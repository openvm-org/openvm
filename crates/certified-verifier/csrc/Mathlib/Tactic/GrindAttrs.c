// Lean compiler output
// Module: Mathlib.Tactic.GrindAttrs
// Imports: public import Init public meta import Init public import Lean.Meta.Tactic.Grind.RegisterCommand public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Attr_grindMod;
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Grind_registerAttr(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "compactness"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 181, 137, 198, 120, 255, 88, 180)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "closedness"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__2_value),LEAN_SCALAR_PTR_LITERAL(192, 112, 91, 214, 208, 182, 66, 163)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__4;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__5;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__6;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__7;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ext"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(241, 12, 90, 240, 78, 252, 149, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(92, 126, 46, 149, 51, 125, 16, 48)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__4_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__4_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__4_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__5_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__4_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(237, 143, 4, 212, 196, 7, 65, 73)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__5_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__5_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__6_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__6_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__6_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__7_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__5_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__6_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(84, 15, 245, 124, 246, 224, 156, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__7_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__7_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__8_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "GrindAttrs"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__8_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__8_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__9_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__7_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__8_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(47, 198, 158, 123, 204, 153, 82, 230)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__9_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__9_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__10_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__9_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),((lean_object*)(((size_t)(943899233) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(30, 70, 63, 115, 230, 122, 160, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__10_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__10_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__11_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__11_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__11_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__12_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__10_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__11_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(9, 213, 153, 225, 118, 156, 120, 133)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__12_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__12_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__13_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__13_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__13_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__14_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__12_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__13_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(33, 191, 126, 230, 127, 105, 219, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__14_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__14_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__15_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__14_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(16, 154, 177, 236, 234, 103, 171, 118)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__15_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__15_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_3_;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(53, 132, 235, 236, 202, 55, 90, 108)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__3 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__3_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__4 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__5 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__5_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__6 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__7 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__7_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__7_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__8 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__8_value;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__9 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__9_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__10 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__11 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__11_value;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__12;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__13;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__14;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness___closed__15;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Attr_compactness;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "compactness!"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(28, 6, 66, 19, 12, 137, 105, 15)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "compactness\?"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(233, 242, 32, 56, 153, 34, 18, 64)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x3f;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "compactness!\?"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(191, 128, 124, 223, 125, 208, 162, 251)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f;
static const lean_string_object lp_mathlib_compactnessTac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "compactnessTac"};
static const lean_object* lp_mathlib_compactnessTac___closed__0 = (const lean_object*)&lp_mathlib_compactnessTac___closed__0_value;
static const lean_ctor_object lp_mathlib_compactnessTac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_compactnessTac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 36, 206, 200, 80, 57, 104, 162)}};
static const lean_object* lp_mathlib_compactnessTac___closed__1 = (const lean_object*)&lp_mathlib_compactnessTac___closed__1_value;
static lean_once_cell_t lp_mathlib_compactnessTac___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_compactnessTac___closed__2;
static lean_once_cell_t lp_mathlib_compactnessTac___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_compactnessTac___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_compactnessTac;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "grind"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__6_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(150, 98, 0, 78, 28, 79, 28, 100)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "grindParam"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__6_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__6_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(16, 144, 208, 205, 52, 106, 220, 83)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "grindLemma"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__6_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(185, 180, 24, 243, 113, 54, 79, 133)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tacticCompactness_x3f___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "tacticCompactness\?_"};
static const lean_object* lp_mathlib_tacticCompactness_x3f___00__closed__0 = (const lean_object*)&lp_mathlib_tacticCompactness_x3f___00__closed__0_value;
static const lean_ctor_object lp_mathlib_tacticCompactness_x3f___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_tacticCompactness_x3f___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(210, 139, 130, 60, 2, 5, 222, 203)}};
static const lean_object* lp_mathlib_tacticCompactness_x3f___00__closed__1 = (const lean_object*)&lp_mathlib_tacticCompactness_x3f___00__closed__1_value;
static lean_once_cell_t lp_mathlib_tacticCompactness_x3f___00__closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticCompactness_x3f___00__closed__2;
static lean_once_cell_t lp_mathlib_tacticCompactness_x3f___00__closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticCompactness_x3f___00__closed__3;
LEAN_EXPORT lean_object* lp_mathlib_tacticCompactness_x3f__;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "grindTrace"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__6_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 245, 239, 189, 189, 113, 196, 115)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "grind\?"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness___closed__0_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__2_value),LEAN_SCALAR_PTR_LITERAL(44, 8, 234, 34, 232, 41, 72, 102)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_closedness___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_closedness___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_closedness___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_closedness___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Attr_closedness;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "closedness!"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(105, 109, 143, 192, 209, 22, 44, 42)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "closedness\?"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(174, 250, 246, 101, 93, 9, 163, 36)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x3f;
static const lean_string_object lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "closedness!\?"};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Lean_Parser_Attr_compactness___closed__2_value),LEAN_SCALAR_PTR_LITERAL(7, 175, 252, 195, 22, 42, 161, 63)}};
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(15, 166, 143, 224, 113, 242, 84, 14)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f;
static const lean_string_object lp_mathlib_closednessTac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "closednessTac"};
static const lean_object* lp_mathlib_closednessTac___closed__0 = (const lean_object*)&lp_mathlib_closednessTac___closed__0_value;
static const lean_ctor_object lp_mathlib_closednessTac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_closednessTac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(191, 22, 179, 245, 185, 96, 252, 70)}};
static const lean_object* lp_mathlib_closednessTac___closed__1 = (const lean_object*)&lp_mathlib_closednessTac___closed__1_value;
static lean_once_cell_t lp_mathlib_closednessTac___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_closednessTac___closed__2;
static lean_once_cell_t lp_mathlib_closednessTac___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_closednessTac___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_closednessTac;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tacticClosedness_x3f___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticClosedness\?_"};
static const lean_object* lp_mathlib_tacticClosedness_x3f___00__closed__0 = (const lean_object*)&lp_mathlib_tacticClosedness_x3f___00__closed__0_value;
static const lean_ctor_object lp_mathlib_tacticClosedness_x3f___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_tacticClosedness_x3f___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 146, 192, 102, 97, 76, 223, 194)}};
static const lean_object* lp_mathlib_tacticClosedness_x3f___00__closed__1 = (const lean_object*)&lp_mathlib_tacticClosedness_x3f___00__closed__1_value;
static lean_once_cell_t lp_mathlib_tacticClosedness_x3f___00__closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticClosedness_x3f___00__closed__2;
static lean_once_cell_t lp_mathlib_tacticClosedness_x3f___00__closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticClosedness_x3f___00__closed__3;
LEAN_EXPORT lean_object* lp_mathlib_tacticClosedness_x3f__;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticClosedness_x3f____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticClosedness_x3f____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2_spec__3___redArg(lean_object* v_x_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
return v_x_1_;
}
else
{
lean_object* v_key_3_; lean_object* v_value_4_; lean_object* v_tail_5_; lean_object* v___x_7_; uint8_t v_isShared_8_; uint8_t v_isSharedCheck_31_; 
v_key_3_ = lean_ctor_get(v_x_2_, 0);
v_value_4_ = lean_ctor_get(v_x_2_, 1);
v_tail_5_ = lean_ctor_get(v_x_2_, 2);
v_isSharedCheck_31_ = !lean_is_exclusive(v_x_2_);
if (v_isSharedCheck_31_ == 0)
{
v___x_7_ = v_x_2_;
v_isShared_8_ = v_isSharedCheck_31_;
goto v_resetjp_6_;
}
else
{
lean_inc(v_tail_5_);
lean_inc(v_value_4_);
lean_inc(v_key_3_);
lean_dec(v_x_2_);
v___x_7_ = lean_box(0);
v_isShared_8_ = v_isSharedCheck_31_;
goto v_resetjp_6_;
}
v_resetjp_6_:
{
lean_object* v___x_9_; uint64_t v___y_11_; 
v___x_9_ = lean_array_get_size(v_x_1_);
if (lean_obj_tag(v_key_3_) == 0)
{
uint64_t v___x_29_; 
v___x_29_ = 1723ULL;
v___y_11_ = v___x_29_;
goto v___jp_10_;
}
else
{
uint64_t v_hash_30_; 
v_hash_30_ = lean_ctor_get_uint64(v_key_3_, sizeof(void*)*2);
v___y_11_ = v_hash_30_;
goto v___jp_10_;
}
v___jp_10_:
{
uint64_t v___x_12_; uint64_t v___x_13_; uint64_t v_fold_14_; uint64_t v___x_15_; uint64_t v___x_16_; uint64_t v___x_17_; size_t v___x_18_; size_t v___x_19_; size_t v___x_20_; size_t v___x_21_; size_t v___x_22_; lean_object* v___x_23_; lean_object* v___x_25_; 
v___x_12_ = 32ULL;
v___x_13_ = lean_uint64_shift_right(v___y_11_, v___x_12_);
v_fold_14_ = lean_uint64_xor(v___y_11_, v___x_13_);
v___x_15_ = 16ULL;
v___x_16_ = lean_uint64_shift_right(v_fold_14_, v___x_15_);
v___x_17_ = lean_uint64_xor(v_fold_14_, v___x_16_);
v___x_18_ = lean_uint64_to_usize(v___x_17_);
v___x_19_ = lean_usize_of_nat(v___x_9_);
v___x_20_ = ((size_t)1ULL);
v___x_21_ = lean_usize_sub(v___x_19_, v___x_20_);
v___x_22_ = lean_usize_land(v___x_18_, v___x_21_);
v___x_23_ = lean_array_uget_borrowed(v_x_1_, v___x_22_);
lean_inc(v___x_23_);
if (v_isShared_8_ == 0)
{
lean_ctor_set(v___x_7_, 2, v___x_23_);
v___x_25_ = v___x_7_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_28_; 
v_reuseFailAlloc_28_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_28_, 0, v_key_3_);
lean_ctor_set(v_reuseFailAlloc_28_, 1, v_value_4_);
lean_ctor_set(v_reuseFailAlloc_28_, 2, v___x_23_);
v___x_25_ = v_reuseFailAlloc_28_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
lean_object* v___x_26_; 
v___x_26_ = lean_array_uset(v_x_1_, v___x_22_, v___x_25_);
v_x_1_ = v___x_26_;
v_x_2_ = v_tail_5_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2___redArg(lean_object* v_i_32_, lean_object* v_source_33_, lean_object* v_target_34_){
_start:
{
lean_object* v___x_35_; uint8_t v___x_36_; 
v___x_35_ = lean_array_get_size(v_source_33_);
v___x_36_ = lean_nat_dec_lt(v_i_32_, v___x_35_);
if (v___x_36_ == 0)
{
lean_dec_ref(v_source_33_);
lean_dec(v_i_32_);
return v_target_34_;
}
else
{
lean_object* v_es_37_; lean_object* v___x_38_; lean_object* v_source_39_; lean_object* v_target_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v_es_37_ = lean_array_fget(v_source_33_, v_i_32_);
v___x_38_ = lean_box(0);
v_source_39_ = lean_array_fset(v_source_33_, v_i_32_, v___x_38_);
v_target_40_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2_spec__3___redArg(v_target_34_, v_es_37_);
v___x_41_ = lean_unsigned_to_nat(1u);
v___x_42_ = lean_nat_add(v_i_32_, v___x_41_);
lean_dec(v_i_32_);
v_i_32_ = v___x_42_;
v_source_33_ = v_source_39_;
v_target_34_ = v_target_40_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1___redArg(lean_object* v_data_44_){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v_nbuckets_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_45_ = lean_array_get_size(v_data_44_);
v___x_46_ = lean_unsigned_to_nat(2u);
v_nbuckets_47_ = lean_nat_mul(v___x_45_, v___x_46_);
v___x_48_ = lean_unsigned_to_nat(0u);
v___x_49_ = lean_box(0);
v___x_50_ = lean_mk_array(v_nbuckets_47_, v___x_49_);
v___x_51_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2___redArg(v___x_48_, v_data_44_, v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___redArg(lean_object* v_a_52_, lean_object* v_x_53_){
_start:
{
if (lean_obj_tag(v_x_53_) == 0)
{
uint8_t v___x_54_; 
v___x_54_ = 0;
return v___x_54_;
}
else
{
lean_object* v_key_55_; lean_object* v_tail_56_; uint8_t v___x_57_; 
v_key_55_ = lean_ctor_get(v_x_53_, 0);
v_tail_56_ = lean_ctor_get(v_x_53_, 2);
v___x_57_ = lean_name_eq(v_key_55_, v_a_52_);
if (v___x_57_ == 0)
{
v_x_53_ = v_tail_56_;
goto _start;
}
else
{
return v___x_57_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___redArg___boxed(lean_object* v_a_59_, lean_object* v_x_60_){
_start:
{
uint8_t v_res_61_; lean_object* v_r_62_; 
v_res_61_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___redArg(v_a_59_, v_x_60_);
lean_dec(v_x_60_);
lean_dec(v_a_59_);
v_r_62_ = lean_box(v_res_61_);
return v_r_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0___redArg(lean_object* v_m_63_, lean_object* v_a_64_, lean_object* v_b_65_){
_start:
{
lean_object* v_size_66_; lean_object* v_buckets_67_; lean_object* v___x_68_; uint64_t v___y_70_; 
v_size_66_ = lean_ctor_get(v_m_63_, 0);
v_buckets_67_ = lean_ctor_get(v_m_63_, 1);
v___x_68_ = lean_array_get_size(v_buckets_67_);
if (lean_obj_tag(v_a_64_) == 0)
{
uint64_t v___x_107_; 
v___x_107_ = 1723ULL;
v___y_70_ = v___x_107_;
goto v___jp_69_;
}
else
{
uint64_t v_hash_108_; 
v_hash_108_ = lean_ctor_get_uint64(v_a_64_, sizeof(void*)*2);
v___y_70_ = v_hash_108_;
goto v___jp_69_;
}
v___jp_69_:
{
uint64_t v___x_71_; uint64_t v___x_72_; uint64_t v_fold_73_; uint64_t v___x_74_; uint64_t v___x_75_; uint64_t v___x_76_; size_t v___x_77_; size_t v___x_78_; size_t v___x_79_; size_t v___x_80_; size_t v___x_81_; lean_object* v_bkt_82_; uint8_t v___x_83_; 
v___x_71_ = 32ULL;
v___x_72_ = lean_uint64_shift_right(v___y_70_, v___x_71_);
v_fold_73_ = lean_uint64_xor(v___y_70_, v___x_72_);
v___x_74_ = 16ULL;
v___x_75_ = lean_uint64_shift_right(v_fold_73_, v___x_74_);
v___x_76_ = lean_uint64_xor(v_fold_73_, v___x_75_);
v___x_77_ = lean_uint64_to_usize(v___x_76_);
v___x_78_ = lean_usize_of_nat(v___x_68_);
v___x_79_ = ((size_t)1ULL);
v___x_80_ = lean_usize_sub(v___x_78_, v___x_79_);
v___x_81_ = lean_usize_land(v___x_77_, v___x_80_);
v_bkt_82_ = lean_array_uget_borrowed(v_buckets_67_, v___x_81_);
v___x_83_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___redArg(v_a_64_, v_bkt_82_);
if (v___x_83_ == 0)
{
lean_object* v___x_85_; uint8_t v_isShared_86_; uint8_t v_isSharedCheck_104_; 
lean_inc_ref(v_buckets_67_);
lean_inc(v_size_66_);
v_isSharedCheck_104_ = !lean_is_exclusive(v_m_63_);
if (v_isSharedCheck_104_ == 0)
{
lean_object* v_unused_105_; lean_object* v_unused_106_; 
v_unused_105_ = lean_ctor_get(v_m_63_, 1);
lean_dec(v_unused_105_);
v_unused_106_ = lean_ctor_get(v_m_63_, 0);
lean_dec(v_unused_106_);
v___x_85_ = v_m_63_;
v_isShared_86_ = v_isSharedCheck_104_;
goto v_resetjp_84_;
}
else
{
lean_dec(v_m_63_);
v___x_85_ = lean_box(0);
v_isShared_86_ = v_isSharedCheck_104_;
goto v_resetjp_84_;
}
v_resetjp_84_:
{
lean_object* v___x_87_; lean_object* v_size_x27_88_; lean_object* v___x_89_; lean_object* v_buckets_x27_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_87_ = lean_unsigned_to_nat(1u);
v_size_x27_88_ = lean_nat_add(v_size_66_, v___x_87_);
lean_dec(v_size_66_);
lean_inc(v_bkt_82_);
v___x_89_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_89_, 0, v_a_64_);
lean_ctor_set(v___x_89_, 1, v_b_65_);
lean_ctor_set(v___x_89_, 2, v_bkt_82_);
v_buckets_x27_90_ = lean_array_uset(v_buckets_67_, v___x_81_, v___x_89_);
v___x_91_ = lean_unsigned_to_nat(4u);
v___x_92_ = lean_nat_mul(v_size_x27_88_, v___x_91_);
v___x_93_ = lean_unsigned_to_nat(3u);
v___x_94_ = lean_nat_div(v___x_92_, v___x_93_);
lean_dec(v___x_92_);
v___x_95_ = lean_array_get_size(v_buckets_x27_90_);
v___x_96_ = lean_nat_dec_le(v___x_94_, v___x_95_);
lean_dec(v___x_94_);
if (v___x_96_ == 0)
{
lean_object* v_val_97_; lean_object* v___x_99_; 
v_val_97_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1___redArg(v_buckets_x27_90_);
if (v_isShared_86_ == 0)
{
lean_ctor_set(v___x_85_, 1, v_val_97_);
lean_ctor_set(v___x_85_, 0, v_size_x27_88_);
v___x_99_ = v___x_85_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_100_; 
v_reuseFailAlloc_100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_100_, 0, v_size_x27_88_);
lean_ctor_set(v_reuseFailAlloc_100_, 1, v_val_97_);
v___x_99_ = v_reuseFailAlloc_100_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
return v___x_99_;
}
}
else
{
lean_object* v___x_102_; 
if (v_isShared_86_ == 0)
{
lean_ctor_set(v___x_85_, 1, v_buckets_x27_90_);
lean_ctor_set(v___x_85_, 0, v_size_x27_88_);
v___x_102_ = v___x_85_;
goto v_reusejp_101_;
}
else
{
lean_object* v_reuseFailAlloc_103_; 
v_reuseFailAlloc_103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_103_, 0, v_size_x27_88_);
lean_ctor_set(v_reuseFailAlloc_103_, 1, v_buckets_x27_90_);
v___x_102_ = v_reuseFailAlloc_103_;
goto v_reusejp_101_;
}
v_reusejp_101_:
{
return v___x_102_;
}
}
}
}
else
{
lean_dec(v_b_65_);
lean_dec(v_a_64_);
return v_m_63_;
}
}
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__4(void){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_115_ = lean_box(0);
v___x_116_ = lean_unsigned_to_nat(16u);
v___x_117_ = lean_mk_array(v___x_116_, v___x_115_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__5(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_118_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__4, &lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__4);
v___x_119_ = lean_unsigned_to_nat(0u);
v___x_120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_119_);
lean_ctor_set(v___x_120_, 1, v___x_118_);
return v___x_120_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__6(void){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_121_ = lean_box(0);
v___x_122_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__3));
v___x_123_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__5, &lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__5_once, _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__5);
v___x_124_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0___redArg(v___x_123_, v___x_122_, v___x_121_);
return v___x_124_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__7(void){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_125_ = lean_box(0);
v___x_126_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__1));
v___x_127_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__6, &lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__6);
v___x_128_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0___redArg(v___x_127_, v___x_126_, v___x_125_);
return v___x_128_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs(void){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__7, &lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__7);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0(lean_object* v_00_u03b2_130_, lean_object* v_m_131_, lean_object* v_a_132_, lean_object* v_b_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0___redArg(v_m_131_, v_a_132_, v_b_133_);
return v___x_134_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0(lean_object* v_00_u03b2_135_, lean_object* v_a_136_, lean_object* v_x_137_){
_start:
{
uint8_t v___x_138_; 
v___x_138_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___redArg(v_a_136_, v_x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0___boxed(lean_object* v_00_u03b2_139_, lean_object* v_a_140_, lean_object* v_x_141_){
_start:
{
uint8_t v_res_142_; lean_object* v_r_143_; 
v_res_142_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__0(v_00_u03b2_139_, v_a_140_, v_x_141_);
lean_dec(v_x_141_);
lean_dec(v_a_140_);
v_r_143_ = lean_box(v_res_142_);
return v_r_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1(lean_object* v_00_u03b2_144_, lean_object* v_data_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1___redArg(v_data_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_147_, lean_object* v_i_148_, lean_object* v_source_149_, lean_object* v_target_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2___redArg(v_i_148_, v_source_149_, v_target_150_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_152_, lean_object* v_x_153_, lean_object* v_x_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs_spec__0_spec__1_spec__2_spec__3___redArg(v_x_153_, v_x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_191_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__1));
v___x_192_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__15_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_));
v___x_193_ = l_Lean_Meta_Grind_registerAttr(v___x_191_, v___x_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5____boxed(lean_object* v_a_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_();
return v_res_195_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__12(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_218_ = l_Lean_Parser_Attr_grindMod;
v___x_219_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__11));
v___x_220_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_221_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_221_, 0, v___x_220_);
lean_ctor_set(v___x_221_, 1, v___x_219_);
lean_ctor_set(v___x_221_, 2, v___x_218_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_222_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__12, &lp_mathlib_Lean_Parser_Attr_compactness___closed__12_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__12);
v___x_223_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__8));
v___x_224_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_224_, 0, v___x_223_);
lean_ctor_set(v___x_224_, 1, v___x_222_);
return v___x_224_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__14(void){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_225_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__13, &lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13);
v___x_226_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__6));
v___x_227_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_228_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_228_, 0, v___x_227_);
lean_ctor_set(v___x_228_, 1, v___x_226_);
lean_ctor_set(v___x_228_, 2, v___x_225_);
return v___x_228_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__15(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_229_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__14, &lp_mathlib_Lean_Parser_Attr_compactness___closed__14_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__14);
v___x_230_ = lean_unsigned_to_nat(1022u);
v___x_231_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__3));
v___x_232_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_232_, 0, v___x_231_);
lean_ctor_set(v___x_232_, 1, v___x_230_);
lean_ctor_set(v___x_232_, 2, v___x_229_);
return v___x_232_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness(void){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__15, &lp_mathlib_Lean_Parser_Attr_compactness___closed__15_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__15);
return v___x_233_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__3(void){
_start:
{
lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_243_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__13, &lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13);
v___x_244_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__2));
v___x_245_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_246_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_246_, 0, v___x_245_);
lean_ctor_set(v___x_246_, 1, v___x_244_);
lean_ctor_set(v___x_246_, 2, v___x_243_);
return v___x_246_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__4(void){
_start:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
v___x_247_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__3, &lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__3_once, _init_lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__3);
v___x_248_ = lean_unsigned_to_nat(1022u);
v___x_249_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__1));
v___x_250_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_250_, 0, v___x_249_);
lean_ctor_set(v___x_250_, 1, v___x_248_);
lean_ctor_set(v___x_250_, 2, v___x_247_);
return v___x_250_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x21(void){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__4, &lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__4_once, _init_lp_mathlib_Lean_Parser_Attr_compactness_x21___closed__4);
return v___x_251_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__3(void){
_start:
{
lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; 
v___x_261_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__13, &lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13);
v___x_262_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__2));
v___x_263_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_264_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_264_, 0, v___x_263_);
lean_ctor_set(v___x_264_, 1, v___x_262_);
lean_ctor_set(v___x_264_, 2, v___x_261_);
return v___x_264_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__4(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; 
v___x_265_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__3, &lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__3_once, _init_lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__3);
v___x_266_ = lean_unsigned_to_nat(1022u);
v___x_267_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__1));
v___x_268_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_268_, 0, v___x_267_);
lean_ctor_set(v___x_268_, 1, v___x_266_);
lean_ctor_set(v___x_268_, 2, v___x_265_);
return v___x_268_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x3f(void){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__4, &lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__4_once, _init_lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__4);
return v___x_269_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__3(void){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_279_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__13, &lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13);
v___x_280_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__2));
v___x_281_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_282_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_282_, 0, v___x_281_);
lean_ctor_set(v___x_282_, 1, v___x_280_);
lean_ctor_set(v___x_282_, 2, v___x_279_);
return v___x_282_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__4(void){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_283_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__3, &lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__3_once, _init_lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__3);
v___x_284_ = lean_unsigned_to_nat(1022u);
v___x_285_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__1));
v___x_286_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_286_, 0, v___x_285_);
lean_ctor_set(v___x_286_, 1, v___x_284_);
lean_ctor_set(v___x_286_, 2, v___x_283_);
return v___x_286_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f(void){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__4, &lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__4_once, _init_lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f___closed__4);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib_compactnessTac___closed__2(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; 
v___x_291_ = l_Lean_Parser_Tactic_optConfig;
v___x_292_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__6));
v___x_293_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_294_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_294_, 0, v___x_293_);
lean_ctor_set(v___x_294_, 1, v___x_292_);
lean_ctor_set(v___x_294_, 2, v___x_291_);
return v___x_294_;
}
}
static lean_object* _init_lp_mathlib_compactnessTac___closed__3(void){
_start:
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_295_ = lean_obj_once(&lp_mathlib_compactnessTac___closed__2, &lp_mathlib_compactnessTac___closed__2_once, _init_lp_mathlib_compactnessTac___closed__2);
v___x_296_ = lean_unsigned_to_nat(1022u);
v___x_297_ = ((lean_object*)(lp_mathlib_compactnessTac___closed__1));
v___x_298_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_298_, 0, v___x_297_);
lean_ctor_set(v___x_298_, 1, v___x_296_);
lean_ctor_set(v___x_298_, 2, v___x_295_);
return v___x_298_;
}
}
static lean_object* _init_lp_mathlib_compactnessTac(void){
_start:
{
lean_object* v___x_299_; 
v___x_299_ = lean_obj_once(&lp_mathlib_compactnessTac___closed__3, &lp_mathlib_compactnessTac___closed__3_once, _init_lp_mathlib_compactnessTac___closed__3);
return v___x_299_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10(void){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = l_Array_mkArray0(lean_box(0));
return v___x_323_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11(void){
_start:
{
lean_object* v___x_324_; lean_object* v___x_325_; 
v___x_324_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__1));
v___x_325_ = l_Lean_mkIdent(v___x_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1(lean_object* v_x_327_, lean_object* v_a_328_, lean_object* v_a_329_){
_start:
{
lean_object* v___x_330_; uint8_t v___x_331_; 
v___x_330_ = ((lean_object*)(lp_mathlib_compactnessTac___closed__1));
lean_inc(v_x_327_);
v___x_331_ = l_Lean_Syntax_isOfKind(v_x_327_, v___x_330_);
if (v___x_331_ == 0)
{
lean_object* v___x_332_; lean_object* v___x_333_; 
lean_dec(v_x_327_);
v___x_332_ = lean_box(1);
v___x_333_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_333_, 0, v___x_332_);
lean_ctor_set(v___x_333_, 1, v_a_329_);
return v___x_333_;
}
else
{
lean_object* v_ref_334_; lean_object* v___x_335_; lean_object* v___x_336_; uint8_t v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v_ref_334_ = lean_ctor_get(v_a_328_, 5);
v___x_335_ = lean_unsigned_to_nat(1u);
v___x_336_ = l_Lean_Syntax_getArg(v_x_327_, v___x_335_);
lean_dec(v_x_327_);
v___x_337_ = 0;
v___x_338_ = l_Lean_SourceInfo_fromRef(v_ref_334_, v___x_337_);
v___x_339_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__0));
v___x_340_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1));
lean_inc_n(v___x_338_, 10);
v___x_341_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_341_, 0, v___x_338_);
lean_ctor_set(v___x_341_, 1, v___x_339_);
v___x_342_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__3));
v___x_343_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__4));
v___x_344_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_338_);
lean_ctor_set(v___x_344_, 1, v___x_343_);
v___x_345_ = l_Lean_Syntax_node1(v___x_338_, v___x_342_, v___x_344_);
v___x_346_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__5));
v___x_347_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_338_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
v___x_348_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7));
v___x_349_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9));
v___x_350_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10, &lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10_once, _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10);
v___x_351_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_351_, 0, v___x_338_);
lean_ctor_set(v___x_351_, 1, v___x_342_);
lean_ctor_set(v___x_351_, 2, v___x_350_);
v___x_352_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11, &lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11_once, _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11);
lean_inc_ref(v___x_351_);
v___x_353_ = l_Lean_Syntax_node2(v___x_338_, v___x_349_, v___x_351_, v___x_352_);
v___x_354_ = l_Lean_Syntax_node1(v___x_338_, v___x_348_, v___x_353_);
v___x_355_ = l_Lean_Syntax_node1(v___x_338_, v___x_342_, v___x_354_);
v___x_356_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__12));
v___x_357_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_357_, 0, v___x_338_);
lean_ctor_set(v___x_357_, 1, v___x_356_);
v___x_358_ = l_Lean_Syntax_node3(v___x_338_, v___x_342_, v___x_347_, v___x_355_, v___x_357_);
v___x_359_ = l_Lean_Syntax_node5(v___x_338_, v___x_340_, v___x_341_, v___x_336_, v___x_345_, v___x_358_, v___x_351_);
v___x_360_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_360_, 0, v___x_359_);
lean_ctor_set(v___x_360_, 1, v_a_329_);
return v___x_360_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___boxed(lean_object* v_x_361_, lean_object* v_a_362_, lean_object* v_a_363_){
_start:
{
lean_object* v_res_364_; 
v_res_364_ = lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1(v_x_361_, v_a_362_, v_a_363_);
lean_dec_ref(v_a_362_);
return v_res_364_;
}
}
static lean_object* _init_lp_mathlib_tacticCompactness_x3f___00__closed__2(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
v___x_368_ = l_Lean_Parser_Tactic_optConfig;
v___x_369_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness_x3f___closed__2));
v___x_370_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_371_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
lean_ctor_set(v___x_371_, 1, v___x_369_);
lean_ctor_set(v___x_371_, 2, v___x_368_);
return v___x_371_;
}
}
static lean_object* _init_lp_mathlib_tacticCompactness_x3f___00__closed__3(void){
_start:
{
lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_372_ = lean_obj_once(&lp_mathlib_tacticCompactness_x3f___00__closed__2, &lp_mathlib_tacticCompactness_x3f___00__closed__2_once, _init_lp_mathlib_tacticCompactness_x3f___00__closed__2);
v___x_373_ = lean_unsigned_to_nat(1022u);
v___x_374_ = ((lean_object*)(lp_mathlib_tacticCompactness_x3f___00__closed__1));
v___x_375_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v___x_373_);
lean_ctor_set(v___x_375_, 2, v___x_372_);
return v___x_375_;
}
}
static lean_object* _init_lp_mathlib_tacticCompactness_x3f__(void){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lean_obj_once(&lp_mathlib_tacticCompactness_x3f___00__closed__3, &lp_mathlib_tacticCompactness_x3f___00__closed__3_once, _init_lp_mathlib_tacticCompactness_x3f___00__closed__3);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1(lean_object* v_x_384_, lean_object* v_a_385_, lean_object* v_a_386_){
_start:
{
lean_object* v___x_387_; uint8_t v___x_388_; 
v___x_387_ = ((lean_object*)(lp_mathlib_tacticCompactness_x3f___00__closed__1));
lean_inc(v_x_384_);
v___x_388_ = l_Lean_Syntax_isOfKind(v_x_384_, v___x_387_);
if (v___x_388_ == 0)
{
lean_object* v___x_389_; lean_object* v___x_390_; 
lean_dec(v_x_384_);
v___x_389_ = lean_box(1);
v___x_390_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_390_, 0, v___x_389_);
lean_ctor_set(v___x_390_, 1, v_a_386_);
return v___x_390_;
}
else
{
lean_object* v_ref_391_; lean_object* v___x_392_; lean_object* v___x_393_; uint8_t v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; 
v_ref_391_ = lean_ctor_get(v_a_385_, 5);
v___x_392_ = lean_unsigned_to_nat(1u);
v___x_393_ = l_Lean_Syntax_getArg(v_x_384_, v___x_392_);
lean_dec(v_x_384_);
v___x_394_ = 0;
v___x_395_ = l_Lean_SourceInfo_fromRef(v_ref_391_, v___x_394_);
v___x_396_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1));
v___x_397_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__2));
lean_inc_n(v___x_395_, 10);
v___x_398_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_398_, 0, v___x_395_);
lean_ctor_set(v___x_398_, 1, v___x_397_);
v___x_399_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__3));
v___x_400_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__4));
v___x_401_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_401_, 0, v___x_395_);
lean_ctor_set(v___x_401_, 1, v___x_400_);
v___x_402_ = l_Lean_Syntax_node1(v___x_395_, v___x_399_, v___x_401_);
v___x_403_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__5));
v___x_404_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_404_, 0, v___x_395_);
lean_ctor_set(v___x_404_, 1, v___x_403_);
v___x_405_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7));
v___x_406_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9));
v___x_407_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10, &lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10_once, _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10);
v___x_408_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_408_, 0, v___x_395_);
lean_ctor_set(v___x_408_, 1, v___x_399_);
lean_ctor_set(v___x_408_, 2, v___x_407_);
v___x_409_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11, &lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11_once, _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__11);
v___x_410_ = l_Lean_Syntax_node2(v___x_395_, v___x_406_, v___x_408_, v___x_409_);
v___x_411_ = l_Lean_Syntax_node1(v___x_395_, v___x_405_, v___x_410_);
v___x_412_ = l_Lean_Syntax_node1(v___x_395_, v___x_399_, v___x_411_);
v___x_413_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__12));
v___x_414_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_414_, 0, v___x_395_);
lean_ctor_set(v___x_414_, 1, v___x_413_);
v___x_415_ = l_Lean_Syntax_node3(v___x_395_, v___x_399_, v___x_404_, v___x_412_, v___x_414_);
v___x_416_ = l_Lean_Syntax_node4(v___x_395_, v___x_396_, v___x_398_, v___x_393_, v___x_402_, v___x_415_);
v___x_417_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_417_, 0, v___x_416_);
lean_ctor_set(v___x_417_, 1, v_a_386_);
return v___x_417_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___boxed(lean_object* v_x_418_, lean_object* v_a_419_, lean_object* v_a_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1(v_x_418_, v_a_419_, v_a_420_);
lean_dec_ref(v_a_419_);
return v_res_421_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; 
v___x_422_ = lean_unsigned_to_nat(2614774725u);
v___x_423_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__9_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_));
v___x_424_ = l_Lean_Name_num___override(v___x_423_, v___x_422_);
return v___x_424_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; 
v___x_425_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__11_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_));
v___x_426_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__0_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_);
v___x_427_ = l_Lean_Name_str___override(v___x_426_, v___x_425_);
return v___x_427_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_428_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__13_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_));
v___x_429_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__1_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_);
v___x_430_ = l_Lean_Name_str___override(v___x_429_, v___x_428_);
return v___x_430_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_431_ = lean_unsigned_to_nat(3u);
v___x_432_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__2_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_);
v___x_433_ = l_Lean_Name_num___override(v___x_432_, v___x_431_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; 
v___x_435_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__3));
v___x_436_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn___closed__3_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_);
v___x_437_ = l_Lean_Meta_Grind_registerAttr(v___x_435_, v___x_436_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5____boxed(lean_object* v_a_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_();
return v_res_439_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness___closed__2(void){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_448_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__13, &lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13);
v___x_449_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness___closed__1));
v___x_450_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_451_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_451_, 0, v___x_450_);
lean_ctor_set(v___x_451_, 1, v___x_449_);
lean_ctor_set(v___x_451_, 2, v___x_448_);
return v___x_451_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness___closed__3(void){
_start:
{
lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_452_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_closedness___closed__2, &lp_mathlib_Lean_Parser_Attr_closedness___closed__2_once, _init_lp_mathlib_Lean_Parser_Attr_closedness___closed__2);
v___x_453_ = lean_unsigned_to_nat(1022u);
v___x_454_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness___closed__0));
v___x_455_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_455_, 0, v___x_454_);
lean_ctor_set(v___x_455_, 1, v___x_453_);
lean_ctor_set(v___x_455_, 2, v___x_452_);
return v___x_455_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness(void){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_closedness___closed__3, &lp_mathlib_Lean_Parser_Attr_closedness___closed__3_once, _init_lp_mathlib_Lean_Parser_Attr_closedness___closed__3);
return v___x_456_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__3(void){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_466_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__13, &lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13);
v___x_467_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__2));
v___x_468_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_469_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_469_, 0, v___x_468_);
lean_ctor_set(v___x_469_, 1, v___x_467_);
lean_ctor_set(v___x_469_, 2, v___x_466_);
return v___x_469_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__4(void){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_470_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__3, &lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__3_once, _init_lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__3);
v___x_471_ = lean_unsigned_to_nat(1022u);
v___x_472_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__1));
v___x_473_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_473_, 0, v___x_472_);
lean_ctor_set(v___x_473_, 1, v___x_471_);
lean_ctor_set(v___x_473_, 2, v___x_470_);
return v___x_473_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x21(void){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__4, &lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__4_once, _init_lp_mathlib_Lean_Parser_Attr_closedness_x21___closed__4);
return v___x_474_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__3(void){
_start:
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; 
v___x_484_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__13, &lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13);
v___x_485_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__2));
v___x_486_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_487_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_487_, 0, v___x_486_);
lean_ctor_set(v___x_487_, 1, v___x_485_);
lean_ctor_set(v___x_487_, 2, v___x_484_);
return v___x_487_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__4(void){
_start:
{
lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; 
v___x_488_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__3, &lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__3_once, _init_lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__3);
v___x_489_ = lean_unsigned_to_nat(1022u);
v___x_490_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__1));
v___x_491_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_491_, 0, v___x_490_);
lean_ctor_set(v___x_491_, 1, v___x_489_);
lean_ctor_set(v___x_491_, 2, v___x_488_);
return v___x_491_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x3f(void){
_start:
{
lean_object* v___x_492_; 
v___x_492_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__4, &lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__4_once, _init_lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__4);
return v___x_492_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__3(void){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; 
v___x_502_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_compactness___closed__13, &lp_mathlib_Lean_Parser_Attr_compactness___closed__13_once, _init_lp_mathlib_Lean_Parser_Attr_compactness___closed__13);
v___x_503_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__2));
v___x_504_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_505_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_505_, 0, v___x_504_);
lean_ctor_set(v___x_505_, 1, v___x_503_);
lean_ctor_set(v___x_505_, 2, v___x_502_);
return v___x_505_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__4(void){
_start:
{
lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; 
v___x_506_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__3, &lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__3_once, _init_lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__3);
v___x_507_ = lean_unsigned_to_nat(1022u);
v___x_508_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__1));
v___x_509_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_509_, 0, v___x_508_);
lean_ctor_set(v___x_509_, 1, v___x_507_);
lean_ctor_set(v___x_509_, 2, v___x_506_);
return v___x_509_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f(void){
_start:
{
lean_object* v___x_510_; 
v___x_510_ = lean_obj_once(&lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__4, &lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__4_once, _init_lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f___closed__4);
return v___x_510_;
}
}
static lean_object* _init_lp_mathlib_closednessTac___closed__2(void){
_start:
{
lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; 
v___x_514_ = l_Lean_Parser_Tactic_optConfig;
v___x_515_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness___closed__1));
v___x_516_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_517_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_517_, 0, v___x_516_);
lean_ctor_set(v___x_517_, 1, v___x_515_);
lean_ctor_set(v___x_517_, 2, v___x_514_);
return v___x_517_;
}
}
static lean_object* _init_lp_mathlib_closednessTac___closed__3(void){
_start:
{
lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; 
v___x_518_ = lean_obj_once(&lp_mathlib_closednessTac___closed__2, &lp_mathlib_closednessTac___closed__2_once, _init_lp_mathlib_closednessTac___closed__2);
v___x_519_ = lean_unsigned_to_nat(1022u);
v___x_520_ = ((lean_object*)(lp_mathlib_closednessTac___closed__1));
v___x_521_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_521_, 0, v___x_520_);
lean_ctor_set(v___x_521_, 1, v___x_519_);
lean_ctor_set(v___x_521_, 2, v___x_518_);
return v___x_521_;
}
}
static lean_object* _init_lp_mathlib_closednessTac(void){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lean_obj_once(&lp_mathlib_closednessTac___closed__3, &lp_mathlib_closednessTac___closed__3_once, _init_lp_mathlib_closednessTac___closed__3);
return v___x_522_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0(void){
_start:
{
lean_object* v___x_523_; lean_object* v___x_524_; 
v___x_523_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs___closed__3));
v___x_524_ = l_Lean_mkIdent(v___x_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1(lean_object* v_x_525_, lean_object* v_a_526_, lean_object* v_a_527_){
_start:
{
lean_object* v___x_528_; uint8_t v___x_529_; 
v___x_528_ = ((lean_object*)(lp_mathlib_closednessTac___closed__1));
lean_inc(v_x_525_);
v___x_529_ = l_Lean_Syntax_isOfKind(v_x_525_, v___x_528_);
if (v___x_529_ == 0)
{
lean_object* v___x_530_; lean_object* v___x_531_; 
lean_dec(v_x_525_);
v___x_530_ = lean_box(1);
v___x_531_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_530_);
lean_ctor_set(v___x_531_, 1, v_a_527_);
return v___x_531_;
}
else
{
lean_object* v_ref_532_; lean_object* v___x_533_; lean_object* v___x_534_; uint8_t v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; 
v_ref_532_ = lean_ctor_get(v_a_526_, 5);
v___x_533_ = lean_unsigned_to_nat(1u);
v___x_534_ = l_Lean_Syntax_getArg(v_x_525_, v___x_533_);
lean_dec(v_x_525_);
v___x_535_ = 0;
v___x_536_ = l_Lean_SourceInfo_fromRef(v_ref_532_, v___x_535_);
v___x_537_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__0));
v___x_538_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__1));
lean_inc_n(v___x_536_, 10);
v___x_539_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_539_, 0, v___x_536_);
lean_ctor_set(v___x_539_, 1, v___x_537_);
v___x_540_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__3));
v___x_541_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__4));
v___x_542_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_542_, 0, v___x_536_);
lean_ctor_set(v___x_542_, 1, v___x_541_);
v___x_543_ = l_Lean_Syntax_node1(v___x_536_, v___x_540_, v___x_542_);
v___x_544_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__5));
v___x_545_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_545_, 0, v___x_536_);
lean_ctor_set(v___x_545_, 1, v___x_544_);
v___x_546_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7));
v___x_547_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9));
v___x_548_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10, &lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10_once, _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10);
v___x_549_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_549_, 0, v___x_536_);
lean_ctor_set(v___x_549_, 1, v___x_540_);
lean_ctor_set(v___x_549_, 2, v___x_548_);
v___x_550_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0, &lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0_once, _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0);
lean_inc_ref(v___x_549_);
v___x_551_ = l_Lean_Syntax_node2(v___x_536_, v___x_547_, v___x_549_, v___x_550_);
v___x_552_ = l_Lean_Syntax_node1(v___x_536_, v___x_546_, v___x_551_);
v___x_553_ = l_Lean_Syntax_node1(v___x_536_, v___x_540_, v___x_552_);
v___x_554_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__12));
v___x_555_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_555_, 0, v___x_536_);
lean_ctor_set(v___x_555_, 1, v___x_554_);
v___x_556_ = l_Lean_Syntax_node3(v___x_536_, v___x_540_, v___x_545_, v___x_553_, v___x_555_);
v___x_557_ = l_Lean_Syntax_node5(v___x_536_, v___x_538_, v___x_539_, v___x_534_, v___x_543_, v___x_556_, v___x_549_);
v___x_558_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_558_, 0, v___x_557_);
lean_ctor_set(v___x_558_, 1, v_a_527_);
return v___x_558_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___boxed(lean_object* v_x_559_, lean_object* v_a_560_, lean_object* v_a_561_){
_start:
{
lean_object* v_res_562_; 
v_res_562_ = lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1(v_x_559_, v_a_560_, v_a_561_);
lean_dec_ref(v_a_560_);
return v_res_562_;
}
}
static lean_object* _init_lp_mathlib_tacticClosedness_x3f___00__closed__2(void){
_start:
{
lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
v___x_566_ = l_Lean_Parser_Tactic_optConfig;
v___x_567_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_closedness_x3f___closed__2));
v___x_568_ = ((lean_object*)(lp_mathlib_Lean_Parser_Attr_compactness___closed__5));
v___x_569_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_569_, 0, v___x_568_);
lean_ctor_set(v___x_569_, 1, v___x_567_);
lean_ctor_set(v___x_569_, 2, v___x_566_);
return v___x_569_;
}
}
static lean_object* _init_lp_mathlib_tacticClosedness_x3f___00__closed__3(void){
_start:
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
v___x_570_ = lean_obj_once(&lp_mathlib_tacticClosedness_x3f___00__closed__2, &lp_mathlib_tacticClosedness_x3f___00__closed__2_once, _init_lp_mathlib_tacticClosedness_x3f___00__closed__2);
v___x_571_ = lean_unsigned_to_nat(1022u);
v___x_572_ = ((lean_object*)(lp_mathlib_tacticClosedness_x3f___00__closed__1));
v___x_573_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
lean_ctor_set(v___x_573_, 1, v___x_571_);
lean_ctor_set(v___x_573_, 2, v___x_570_);
return v___x_573_;
}
}
static lean_object* _init_lp_mathlib_tacticClosedness_x3f__(void){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lean_obj_once(&lp_mathlib_tacticClosedness_x3f___00__closed__3, &lp_mathlib_tacticClosedness_x3f___00__closed__3_once, _init_lp_mathlib_tacticClosedness_x3f___00__closed__3);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticClosedness_x3f____1(lean_object* v_x_575_, lean_object* v_a_576_, lean_object* v_a_577_){
_start:
{
lean_object* v___x_578_; uint8_t v___x_579_; 
v___x_578_ = ((lean_object*)(lp_mathlib_tacticClosedness_x3f___00__closed__1));
lean_inc(v_x_575_);
v___x_579_ = l_Lean_Syntax_isOfKind(v_x_575_, v___x_578_);
if (v___x_579_ == 0)
{
lean_object* v___x_580_; lean_object* v___x_581_; 
lean_dec(v_x_575_);
v___x_580_ = lean_box(1);
v___x_581_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_581_, 0, v___x_580_);
lean_ctor_set(v___x_581_, 1, v_a_577_);
return v___x_581_;
}
else
{
lean_object* v_ref_582_; lean_object* v___x_583_; lean_object* v___x_584_; uint8_t v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
v_ref_582_ = lean_ctor_get(v_a_576_, 5);
v___x_583_ = lean_unsigned_to_nat(1u);
v___x_584_ = l_Lean_Syntax_getArg(v_x_575_, v___x_583_);
lean_dec(v_x_575_);
v___x_585_ = 0;
v___x_586_ = l_Lean_SourceInfo_fromRef(v_ref_582_, v___x_585_);
v___x_587_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__1));
v___x_588_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticCompactness_x3f____1___closed__2));
lean_inc_n(v___x_586_, 10);
v___x_589_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_589_, 0, v___x_586_);
lean_ctor_set(v___x_589_, 1, v___x_588_);
v___x_590_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__3));
v___x_591_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__4));
v___x_592_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_592_, 0, v___x_586_);
lean_ctor_set(v___x_592_, 1, v___x_591_);
v___x_593_ = l_Lean_Syntax_node1(v___x_586_, v___x_590_, v___x_592_);
v___x_594_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__5));
v___x_595_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_595_, 0, v___x_586_);
lean_ctor_set(v___x_595_, 1, v___x_594_);
v___x_596_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__7));
v___x_597_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__9));
v___x_598_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10, &lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10_once, _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__10);
v___x_599_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_599_, 0, v___x_586_);
lean_ctor_set(v___x_599_, 1, v___x_590_);
lean_ctor_set(v___x_599_, 2, v___x_598_);
v___x_600_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0, &lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0_once, _init_lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__closednessTac__1___closed__0);
v___x_601_ = l_Lean_Syntax_node2(v___x_586_, v___x_597_, v___x_599_, v___x_600_);
v___x_602_ = l_Lean_Syntax_node1(v___x_586_, v___x_596_, v___x_601_);
v___x_603_ = l_Lean_Syntax_node1(v___x_586_, v___x_590_, v___x_602_);
v___x_604_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__compactnessTac__1___closed__12));
v___x_605_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_586_);
lean_ctor_set(v___x_605_, 1, v___x_604_);
v___x_606_ = l_Lean_Syntax_node3(v___x_586_, v___x_590_, v___x_595_, v___x_603_, v___x_605_);
v___x_607_ = l_Lean_Syntax_node4(v___x_586_, v___x_587_, v___x_589_, v___x_584_, v___x_593_, v___x_606_);
v___x_608_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_608_, 0, v___x_607_);
lean_ctor_set(v___x_608_, 1, v_a_577_);
return v___x_608_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticClosedness_x3f____1___boxed(lean_object* v_x_609_, lean_object* v_a_610_, lean_object* v_a_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_mathlib___aux__Mathlib__Tactic__GrindAttrs______macroRules__tacticClosedness_x3f____1(v_x_609_, v_a_610_, v_a_611_);
lean_dec_ref(v_a_610_);
return v_res_612_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Grind_RegisterCommand(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_GrindAttrs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Grind_RegisterCommand(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs = _init_lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__Mathlib_grindAttrs);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_GrindAttrs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_GrindAttrs_943899233____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Lean_Parser_Attr_compactness = _init_lp_mathlib_Lean_Parser_Attr_compactness();
lean_mark_persistent(lp_mathlib_Lean_Parser_Attr_compactness);
lp_mathlib_Lean_Parser_Attr_compactness_x21 = _init_lp_mathlib_Lean_Parser_Attr_compactness_x21();
lean_mark_persistent(lp_mathlib_Lean_Parser_Attr_compactness_x21);
lp_mathlib_Lean_Parser_Attr_compactness_x3f = _init_lp_mathlib_Lean_Parser_Attr_compactness_x3f();
lean_mark_persistent(lp_mathlib_Lean_Parser_Attr_compactness_x3f);
lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f = _init_lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f();
lean_mark_persistent(lp_mathlib_Lean_Parser_Attr_compactness_x21_x3f);
lp_mathlib_compactnessTac = _init_lp_mathlib_compactnessTac();
lean_mark_persistent(lp_mathlib_compactnessTac);
lp_mathlib_tacticCompactness_x3f__ = _init_lp_mathlib_tacticCompactness_x3f__();
lean_mark_persistent(lp_mathlib_tacticCompactness_x3f__);
res = lp_mathlib___private_Mathlib_Tactic_GrindAttrs_0__initFn_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_GrindAttrs_2614774725____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Lean_Parser_Attr_closedness = _init_lp_mathlib_Lean_Parser_Attr_closedness();
lean_mark_persistent(lp_mathlib_Lean_Parser_Attr_closedness);
lp_mathlib_Lean_Parser_Attr_closedness_x21 = _init_lp_mathlib_Lean_Parser_Attr_closedness_x21();
lean_mark_persistent(lp_mathlib_Lean_Parser_Attr_closedness_x21);
lp_mathlib_Lean_Parser_Attr_closedness_x3f = _init_lp_mathlib_Lean_Parser_Attr_closedness_x3f();
lean_mark_persistent(lp_mathlib_Lean_Parser_Attr_closedness_x3f);
lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f = _init_lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f();
lean_mark_persistent(lp_mathlib_Lean_Parser_Attr_closedness_x21_x3f);
lp_mathlib_closednessTac = _init_lp_mathlib_closednessTac();
lean_mark_persistent(lp_mathlib_closednessTac);
lp_mathlib_tacticClosedness_x3f__ = _init_lp_mathlib_tacticClosedness_x3f__();
lean_mark_persistent(lp_mathlib_tacticClosedness_x3f__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Grind_RegisterCommand(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_GrindAttrs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Grind_RegisterCommand(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_GrindAttrs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_GrindAttrs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_GrindAttrs(builtin);
}
#ifdef __cplusplus
}
#endif
