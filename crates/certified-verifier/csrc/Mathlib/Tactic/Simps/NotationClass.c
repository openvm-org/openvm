// Lean compiler output
// Module: Mathlib.Tactic.Simps.NotationClass
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Exception public meta import Batteries.Lean.NameMapAttribute
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_mkRawNatLit(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_getStructureFields(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
uint8_t l_Lean_isStructure(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getNumHeadForalls(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Lean_Environment_findConstVal_x3f(lean_object*, lean_object*, uint8_t);
lean_object* lp_batteries_Lean_registerNameMapExtension___redArg(lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_mkLevelParam(lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_notation__class___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "notation_class"};
static const lean_object* lp_mathlib_notation__class___closed__0 = (const lean_object*)&lp_mathlib_notation__class___closed__0_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_notation__class___closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 101, 62, 56, 194, 190, 116, 110)}};
static const lean_object* lp_mathlib_notation__class___closed__1 = (const lean_object*)&lp_mathlib_notation__class___closed__1_value;
static const lean_string_object lp_mathlib_notation__class___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_notation__class___closed__2 = (const lean_object*)&lp_mathlib_notation__class___closed__2_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_notation__class___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_notation__class___closed__3 = (const lean_object*)&lp_mathlib_notation__class___closed__3_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_notation__class___closed__4 = (const lean_object*)&lp_mathlib_notation__class___closed__4_value;
static const lean_string_object lp_mathlib_notation__class___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_notation__class___closed__5 = (const lean_object*)&lp_mathlib_notation__class___closed__5_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_notation__class___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_notation__class___closed__6 = (const lean_object*)&lp_mathlib_notation__class___closed__6_value;
static const lean_string_object lp_mathlib_notation__class___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_mathlib_notation__class___closed__7 = (const lean_object*)&lp_mathlib_notation__class___closed__7_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__7_value)}};
static const lean_object* lp_mathlib_notation__class___closed__8 = (const lean_object*)&lp_mathlib_notation__class___closed__8_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__6_value),((lean_object*)&lp_mathlib_notation__class___closed__8_value)}};
static const lean_object* lp_mathlib_notation__class___closed__9 = (const lean_object*)&lp_mathlib_notation__class___closed__9_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__3_value),((lean_object*)&lp_mathlib_notation__class___closed__4_value),((lean_object*)&lp_mathlib_notation__class___closed__9_value)}};
static const lean_object* lp_mathlib_notation__class___closed__10 = (const lean_object*)&lp_mathlib_notation__class___closed__10_value;
static const lean_string_object lp_mathlib_notation__class___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_notation__class___closed__11 = (const lean_object*)&lp_mathlib_notation__class___closed__11_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_notation__class___closed__11_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_notation__class___closed__12 = (const lean_object*)&lp_mathlib_notation__class___closed__12_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__12_value)}};
static const lean_object* lp_mathlib_notation__class___closed__13 = (const lean_object*)&lp_mathlib_notation__class___closed__13_value;
static const lean_string_object lp_mathlib_notation__class___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_notation__class___closed__14 = (const lean_object*)&lp_mathlib_notation__class___closed__14_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_notation__class___closed__14_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_notation__class___closed__15 = (const lean_object*)&lp_mathlib_notation__class___closed__15_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__15_value)}};
static const lean_object* lp_mathlib_notation__class___closed__16 = (const lean_object*)&lp_mathlib_notation__class___closed__16_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__3_value),((lean_object*)&lp_mathlib_notation__class___closed__13_value),((lean_object*)&lp_mathlib_notation__class___closed__16_value)}};
static const lean_object* lp_mathlib_notation__class___closed__17 = (const lean_object*)&lp_mathlib_notation__class___closed__17_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__6_value),((lean_object*)&lp_mathlib_notation__class___closed__17_value)}};
static const lean_object* lp_mathlib_notation__class___closed__18 = (const lean_object*)&lp_mathlib_notation__class___closed__18_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__3_value),((lean_object*)&lp_mathlib_notation__class___closed__10_value),((lean_object*)&lp_mathlib_notation__class___closed__18_value)}};
static const lean_object* lp_mathlib_notation__class___closed__19 = (const lean_object*)&lp_mathlib_notation__class___closed__19_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__3_value),((lean_object*)&lp_mathlib_notation__class___closed__19_value),((lean_object*)&lp_mathlib_notation__class___closed__18_value)}};
static const lean_object* lp_mathlib_notation__class___closed__20 = (const lean_object*)&lp_mathlib_notation__class___closed__20_value;
static const lean_ctor_object lp_mathlib_notation__class___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_notation__class___closed__20_value)}};
static const lean_object* lp_mathlib_notation__class___closed__21 = (const lean_object*)&lp_mathlib_notation__class___closed__21_value;
LEAN_EXPORT const lean_object* lp_mathlib_notation__class = (const lean_object*)&lp_mathlib_notation__class___closed__21_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Simps_defaultfindArgs_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Simps_defaultfindArgs_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Simps_defaultfindArgs___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 75, .m_data = "initialize_simps_projections cannot automatically find arguments for class "};
static const lean_object* lp_mathlib_Simps_defaultfindArgs___redArg___closed__0 = (const lean_object*)&lp_mathlib_Simps_defaultfindArgs___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Simps_defaultfindArgs___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_defaultfindArgs___redArg___closed__1;
static const lean_string_object lp_mathlib_Simps_defaultfindArgs___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "no such class "};
static const lean_object* lp_mathlib_Simps_defaultfindArgs___redArg___closed__2 = (const lean_object*)&lp_mathlib_Simps_defaultfindArgs___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Simps_defaultfindArgs___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_defaultfindArgs___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Simps_defaultfindArgs___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_defaultfindArgs___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_defaultfindArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_defaultfindArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Simps_copyFirst___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_mathlib_Simps_copyFirst___redArg___closed__0 = (const lean_object*)&lp_mathlib_Simps_copyFirst___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Simps_copyFirst___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Simps_copyFirst___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_mathlib_Simps_copyFirst___redArg___closed__1 = (const lean_object*)&lp_mathlib_Simps_copyFirst___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Simps_copyFirst___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_copyFirst___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Simps_copyFirst___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_copyFirst___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_copyFirst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_copyFirst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_copySecond___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_copySecond___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_copySecond(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_copySecond___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Simps_nsmulArgs___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Simps_nsmulArgs___redArg___closed__0 = (const lean_object*)&lp_mathlib_Simps_nsmulArgs___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Simps_nsmulArgs___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Simps_nsmulArgs___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Simps_nsmulArgs___redArg___closed__1 = (const lean_object*)&lp_mathlib_Simps_nsmulArgs___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Simps_nsmulArgs___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_nsmulArgs___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Simps_nsmulArgs___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_nsmulArgs___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Simps_nsmulArgs___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_nsmulArgs___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_nsmulArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_nsmulArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Simps_zsmulArgs___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Simps_zsmulArgs___redArg___closed__0 = (const lean_object*)&lp_mathlib_Simps_zsmulArgs___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Simps_zsmulArgs___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Simps_zsmulArgs___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_object* lp_mathlib_Simps_zsmulArgs___redArg___closed__1 = (const lean_object*)&lp_mathlib_Simps_zsmulArgs___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Simps_zsmulArgs___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_zsmulArgs___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Simps_zsmulArgs___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_zsmulArgs___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Simps_zsmulArgs___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_zsmulArgs___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_zsmulArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_zsmulArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Simps_findZeroArgs___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_findZeroArgs___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Simps_findZeroArgs___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_findZeroArgs___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Simps_findZeroArgs___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_findZeroArgs___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_findZeroArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_findZeroArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Simps_findOneArgs___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_findOneArgs___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Simps_findOneArgs___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Simps_findOneArgs___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Simps_findOneArgs___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_findOneArgs___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_findOneArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_findOneArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__14 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__14_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__15;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__16 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__16_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__17;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__18 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__18_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_findCoercionArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_findCoercionArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Simps"};
static const lean_object* lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__0 = (const lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__0_value;
static const lean_string_object lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "defaultfindArgs"};
static const lean_object* lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__1 = (const lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__1_value;
static const lean_ctor_object lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 14, 47, 163, 29, 240, 87, 177)}};
static const lean_ctor_object lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__1_value),LEAN_SCALAR_PTR_LITERAL(58, 238, 166, 146, 182, 41, 224, 220)}};
static const lean_object* lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__2 = (const lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__2_value;
static const lean_ctor_object lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__2_value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__3 = (const lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default = (const lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Simps_instInhabitedAutomaticProjectionData = (const lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Already exists entry for "};
static const lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "declaration "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = " has wrong type"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "no such declaration "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "findArgType"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__11_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "@[notation_class] attribute can only be added to classes."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__11_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__11_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__12_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__12_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "notationClassAttr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 14, 47, 163, 29, 240, 87, 177)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(71, 1, 215, 57, 205, 161, 244, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_notation__class___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "An attribute specifying that this is a notation class. Used by @[simps]."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__4_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_notation__class___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__4_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__4_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Simps_notationClassAttr;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1(size_t v_sz_53_, size_t v_i_54_, lean_object* v_bs_55_){
_start:
{
uint8_t v___x_56_; 
v___x_56_ = lean_usize_dec_lt(v_i_54_, v_sz_53_);
if (v___x_56_ == 0)
{
return v_bs_55_;
}
else
{
lean_object* v_v_57_; lean_object* v___x_58_; lean_object* v_bs_x27_59_; lean_object* v___x_60_; size_t v___x_61_; size_t v___x_62_; lean_object* v___x_63_; 
v_v_57_ = lean_array_uget(v_bs_55_, v_i_54_);
v___x_58_ = lean_unsigned_to_nat(0u);
v_bs_x27_59_ = lean_array_uset(v_bs_55_, v_i_54_, v___x_58_);
v___x_60_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_60_, 0, v_v_57_);
v___x_61_ = ((size_t)1ULL);
v___x_62_ = lean_usize_add(v_i_54_, v___x_61_);
v___x_63_ = lean_array_uset(v_bs_x27_59_, v_i_54_, v___x_60_);
v_i_54_ = v___x_62_;
v_bs_55_ = v___x_63_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1___boxed(lean_object* v_sz_65_, lean_object* v_i_66_, lean_object* v_bs_67_){
_start:
{
size_t v_sz_boxed_68_; size_t v_i_boxed_69_; lean_object* v_res_70_; 
v_sz_boxed_68_ = lean_unbox_usize(v_sz_65_);
lean_dec(v_sz_65_);
v_i_boxed_69_ = lean_unbox_usize(v_i_66_);
lean_dec(v_i_66_);
v_res_70_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1(v_sz_boxed_68_, v_i_boxed_69_, v_bs_67_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Simps_defaultfindArgs_spec__0_spec__0(lean_object* v_msgData_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_){
_start:
{
lean_object* v___x_77_; lean_object* v_env_78_; lean_object* v___x_79_; lean_object* v_mctx_80_; lean_object* v_lctx_81_; lean_object* v_options_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_77_ = lean_st_ref_get(v___y_75_);
v_env_78_ = lean_ctor_get(v___x_77_, 0);
lean_inc_ref(v_env_78_);
lean_dec(v___x_77_);
v___x_79_ = lean_st_ref_get(v___y_73_);
v_mctx_80_ = lean_ctor_get(v___x_79_, 0);
lean_inc_ref(v_mctx_80_);
lean_dec(v___x_79_);
v_lctx_81_ = lean_ctor_get(v___y_72_, 2);
v_options_82_ = lean_ctor_get(v___y_74_, 2);
lean_inc_ref(v_options_82_);
lean_inc_ref(v_lctx_81_);
v___x_83_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_83_, 0, v_env_78_);
lean_ctor_set(v___x_83_, 1, v_mctx_80_);
lean_ctor_set(v___x_83_, 2, v_lctx_81_);
lean_ctor_set(v___x_83_, 3, v_options_82_);
v___x_84_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_msgData_71_);
v___x_85_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Simps_defaultfindArgs_spec__0_spec__0___boxed(lean_object* v_msgData_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Simps_defaultfindArgs_spec__0_spec__0(v_msgData_86_, v___y_87_, v___y_88_, v___y_89_, v___y_90_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
lean_dec(v___y_88_);
lean_dec_ref(v___y_87_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg(lean_object* v_msg_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v_ref_99_; lean_object* v___x_100_; lean_object* v_a_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_109_; 
v_ref_99_ = lean_ctor_get(v___y_96_, 5);
v___x_100_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Simps_defaultfindArgs_spec__0_spec__0(v_msg_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
v_a_101_ = lean_ctor_get(v___x_100_, 0);
v_isSharedCheck_109_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_109_ == 0)
{
v___x_103_ = v___x_100_;
v_isShared_104_ = v_isSharedCheck_109_;
goto v_resetjp_102_;
}
else
{
lean_inc(v_a_101_);
lean_dec(v___x_100_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_109_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v___x_105_; lean_object* v___x_107_; 
lean_inc(v_ref_99_);
v___x_105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_105_, 0, v_ref_99_);
lean_ctor_set(v___x_105_, 1, v_a_101_);
if (v_isShared_104_ == 0)
{
lean_ctor_set_tag(v___x_103_, 1);
lean_ctor_set(v___x_103_, 0, v___x_105_);
v___x_107_ = v___x_103_;
goto v_reusejp_106_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v___x_105_);
v___x_107_ = v_reuseFailAlloc_108_;
goto v_reusejp_106_;
}
v_reusejp_106_:
{
return v___x_107_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg___boxed(lean_object* v_msg_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg(v_msg_110_, v___y_111_, v___y_112_, v___y_113_, v___y_114_);
lean_dec(v___y_114_);
lean_dec_ref(v___y_113_);
lean_dec(v___y_112_);
lean_dec_ref(v___y_111_);
return v_res_116_;
}
}
static lean_object* _init_lp_mathlib_Simps_defaultfindArgs___redArg___closed__1(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_118_ = ((lean_object*)(lp_mathlib_Simps_defaultfindArgs___redArg___closed__0));
v___x_119_ = l_Lean_stringToMessageData(v___x_118_);
return v___x_119_;
}
}
static lean_object* _init_lp_mathlib_Simps_defaultfindArgs___redArg___closed__3(void){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_121_ = ((lean_object*)(lp_mathlib_Simps_defaultfindArgs___redArg___closed__2));
v___x_122_ = l_Lean_stringToMessageData(v___x_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_defaultfindArgs___redArg(lean_object* v_className_123_, lean_object* v_args_124_, lean_object* v_a_125_, lean_object* v_a_126_, lean_object* v_a_127_, lean_object* v_a_128_){
_start:
{
lean_object* v___x_130_; lean_object* v_env_131_; uint8_t v___x_132_; lean_object* v___x_133_; 
v___x_130_ = lean_st_ref_get(v_a_128_);
v_env_131_ = lean_ctor_get(v___x_130_, 0);
lean_inc_ref(v_env_131_);
lean_dec(v___x_130_);
v___x_132_ = 0;
lean_inc(v_className_123_);
v___x_133_ = l_Lean_Environment_find_x3f(v_env_131_, v_className_123_, v___x_132_);
if (lean_obj_tag(v___x_133_) == 1)
{
lean_object* v_val_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_159_; 
v_val_134_ = lean_ctor_get(v___x_133_, 0);
v_isSharedCheck_159_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_159_ == 0)
{
v___x_136_ = v___x_133_;
v_isShared_137_ = v_isSharedCheck_159_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_val_134_);
lean_dec(v___x_133_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_159_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; uint8_t v___x_141_; 
v___x_138_ = l_Lean_ConstantInfo_type(v_val_134_);
lean_dec(v_val_134_);
v___x_139_ = l_Lean_Expr_getNumHeadForalls(v___x_138_);
lean_dec_ref(v___x_138_);
v___x_140_ = lean_array_get_size(v_args_124_);
v___x_141_ = lean_nat_dec_eq(v___x_139_, v___x_140_);
if (v___x_141_ == 0)
{
lean_object* v___x_142_; uint8_t v___x_143_; 
v___x_142_ = lean_unsigned_to_nat(1u);
v___x_143_ = lean_nat_dec_eq(v___x_140_, v___x_142_);
if (v___x_143_ == 0)
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
lean_dec(v___x_139_);
lean_del_object(v___x_136_);
lean_dec_ref(v_args_124_);
v___x_144_ = lean_obj_once(&lp_mathlib_Simps_defaultfindArgs___redArg___closed__1, &lp_mathlib_Simps_defaultfindArgs___redArg___closed__1_once, _init_lp_mathlib_Simps_defaultfindArgs___redArg___closed__1);
v___x_145_ = l_Lean_MessageData_ofName(v_className_123_);
v___x_146_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_146_, 0, v___x_144_);
lean_ctor_set(v___x_146_, 1, v___x_145_);
v___x_147_ = lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg(v___x_146_, v_a_125_, v_a_126_, v_a_127_, v_a_128_);
return v___x_147_;
}
else
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_151_; 
lean_dec(v_className_123_);
v___x_148_ = lean_unsigned_to_nat(0u);
v___x_149_ = lean_array_fget(v_args_124_, v___x_148_);
lean_dec_ref(v_args_124_);
if (v_isShared_137_ == 0)
{
lean_ctor_set(v___x_136_, 0, v___x_149_);
v___x_151_ = v___x_136_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v___x_149_);
v___x_151_ = v_reuseFailAlloc_154_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_152_ = lean_mk_array(v___x_139_, v___x_151_);
v___x_153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
return v___x_153_;
}
}
}
else
{
size_t v_sz_155_; size_t v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
lean_dec(v___x_139_);
lean_del_object(v___x_136_);
lean_dec(v_className_123_);
v_sz_155_ = lean_array_size(v_args_124_);
v___x_156_ = ((size_t)0ULL);
v___x_157_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1(v_sz_155_, v___x_156_, v_args_124_);
v___x_158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
return v___x_158_;
}
}
}
else
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
lean_dec(v___x_133_);
lean_dec_ref(v_args_124_);
v___x_160_ = lean_obj_once(&lp_mathlib_Simps_defaultfindArgs___redArg___closed__3, &lp_mathlib_Simps_defaultfindArgs___redArg___closed__3_once, _init_lp_mathlib_Simps_defaultfindArgs___redArg___closed__3);
v___x_161_ = l_Lean_MessageData_ofName(v_className_123_);
v___x_162_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_160_);
lean_ctor_set(v___x_162_, 1, v___x_161_);
v___x_163_ = lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg(v___x_162_, v_a_125_, v_a_126_, v_a_127_, v_a_128_);
return v___x_163_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_defaultfindArgs___redArg___boxed(lean_object* v_className_164_, lean_object* v_args_165_, lean_object* v_a_166_, lean_object* v_a_167_, lean_object* v_a_168_, lean_object* v_a_169_, lean_object* v_a_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_Simps_defaultfindArgs___redArg(v_className_164_, v_args_165_, v_a_166_, v_a_167_, v_a_168_, v_a_169_);
lean_dec(v_a_169_);
lean_dec_ref(v_a_168_);
lean_dec(v_a_167_);
lean_dec_ref(v_a_166_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_defaultfindArgs(lean_object* v_x_172_, lean_object* v_className_173_, lean_object* v_args_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lp_mathlib_Simps_defaultfindArgs___redArg(v_className_173_, v_args_174_, v_a_175_, v_a_176_, v_a_177_, v_a_178_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_defaultfindArgs___boxed(lean_object* v_x_181_, lean_object* v_className_182_, lean_object* v_args_183_, lean_object* v_a_184_, lean_object* v_a_185_, lean_object* v_a_186_, lean_object* v_a_187_, lean_object* v_a_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_Simps_defaultfindArgs(v_x_181_, v_className_182_, v_args_183_, v_a_184_, v_a_185_, v_a_186_, v_a_187_);
lean_dec(v_a_187_);
lean_dec_ref(v_a_186_);
lean_dec(v_a_185_);
lean_dec_ref(v_a_184_);
lean_dec(v_x_181_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0(lean_object* v_00_u03b1_190_, lean_object* v_msg_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg(v_msg_191_, v___y_192_, v___y_193_, v___y_194_, v___y_195_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___boxed(lean_object* v_00_u03b1_198_, lean_object* v_msg_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0(v_00_u03b1_198_, v_msg_199_, v___y_200_, v___y_201_, v___y_202_, v___y_203_);
lean_dec(v___y_203_);
lean_dec_ref(v___y_202_);
lean_dec(v___y_201_);
lean_dec_ref(v___y_200_);
return v_res_205_;
}
}
static lean_object* _init_lp_mathlib_Simps_copyFirst___redArg___closed__2(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_209_ = lean_box(0);
v___x_210_ = ((lean_object*)(lp_mathlib_Simps_copyFirst___redArg___closed__1));
v___x_211_ = l_Lean_Expr_const___override(v___x_210_, v___x_209_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_copyFirst___redArg(lean_object* v_args_212_){
_start:
{
lean_object* v___y_215_; lean_object* v___x_221_; lean_object* v___x_222_; uint8_t v___x_223_; 
v___x_221_ = lean_unsigned_to_nat(0u);
v___x_222_ = lean_array_get_size(v_args_212_);
v___x_223_ = lean_nat_dec_lt(v___x_221_, v___x_222_);
if (v___x_223_ == 0)
{
lean_object* v___x_224_; 
v___x_224_ = lean_obj_once(&lp_mathlib_Simps_copyFirst___redArg___closed__2, &lp_mathlib_Simps_copyFirst___redArg___closed__2_once, _init_lp_mathlib_Simps_copyFirst___redArg___closed__2);
v___y_215_ = v___x_224_;
goto v___jp_214_;
}
else
{
lean_object* v___x_225_; 
v___x_225_ = lean_array_fget_borrowed(v_args_212_, v___x_221_);
lean_inc(v___x_225_);
v___y_215_ = v___x_225_;
goto v___jp_214_;
}
v___jp_214_:
{
lean_object* v___x_216_; size_t v_sz_217_; size_t v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v___x_216_ = lean_array_push(v_args_212_, v___y_215_);
v_sz_217_ = lean_array_size(v___x_216_);
v___x_218_ = ((size_t)0ULL);
v___x_219_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1(v_sz_217_, v___x_218_, v___x_216_);
v___x_220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_220_, 0, v___x_219_);
return v___x_220_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_copyFirst___redArg___boxed(lean_object* v_args_226_, lean_object* v_a_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_Simps_copyFirst___redArg(v_args_226_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_copyFirst(lean_object* v_x_229_, lean_object* v_x_230_, lean_object* v_args_231_, lean_object* v_a_232_, lean_object* v_a_233_, lean_object* v_a_234_, lean_object* v_a_235_){
_start:
{
lean_object* v___x_237_; 
v___x_237_ = lp_mathlib_Simps_copyFirst___redArg(v_args_231_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_copyFirst___boxed(lean_object* v_x_238_, lean_object* v_x_239_, lean_object* v_args_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_, lean_object* v_a_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_Simps_copyFirst(v_x_238_, v_x_239_, v_args_240_, v_a_241_, v_a_242_, v_a_243_, v_a_244_);
lean_dec(v_a_244_);
lean_dec_ref(v_a_243_);
lean_dec(v_a_242_);
lean_dec_ref(v_a_241_);
lean_dec(v_x_239_);
lean_dec(v_x_238_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_copySecond___redArg(lean_object* v_args_247_){
_start:
{
lean_object* v___y_250_; lean_object* v___x_256_; lean_object* v___x_257_; uint8_t v___x_258_; 
v___x_256_ = lean_unsigned_to_nat(1u);
v___x_257_ = lean_array_get_size(v_args_247_);
v___x_258_ = lean_nat_dec_lt(v___x_256_, v___x_257_);
if (v___x_258_ == 0)
{
lean_object* v___x_259_; 
v___x_259_ = lean_obj_once(&lp_mathlib_Simps_copyFirst___redArg___closed__2, &lp_mathlib_Simps_copyFirst___redArg___closed__2_once, _init_lp_mathlib_Simps_copyFirst___redArg___closed__2);
v___y_250_ = v___x_259_;
goto v___jp_249_;
}
else
{
lean_object* v___x_260_; 
v___x_260_ = lean_array_fget_borrowed(v_args_247_, v___x_256_);
lean_inc(v___x_260_);
v___y_250_ = v___x_260_;
goto v___jp_249_;
}
v___jp_249_:
{
lean_object* v___x_251_; size_t v_sz_252_; size_t v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_251_ = lean_array_push(v_args_247_, v___y_250_);
v_sz_252_ = lean_array_size(v___x_251_);
v___x_253_ = ((size_t)0ULL);
v___x_254_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1(v_sz_252_, v___x_253_, v___x_251_);
v___x_255_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
return v___x_255_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_copySecond___redArg___boxed(lean_object* v_args_261_, lean_object* v_a_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_Simps_copySecond___redArg(v_args_261_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_copySecond(lean_object* v_x_264_, lean_object* v_x_265_, lean_object* v_args_266_, lean_object* v_a_267_, lean_object* v_a_268_, lean_object* v_a_269_, lean_object* v_a_270_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_mathlib_Simps_copySecond___redArg(v_args_266_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_copySecond___boxed(lean_object* v_x_273_, lean_object* v_x_274_, lean_object* v_args_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_, lean_object* v_a_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_Simps_copySecond(v_x_273_, v_x_274_, v_args_275_, v_a_276_, v_a_277_, v_a_278_, v_a_279_);
lean_dec(v_a_279_);
lean_dec_ref(v_a_278_);
lean_dec(v_a_277_);
lean_dec_ref(v_a_276_);
lean_dec(v_x_274_);
lean_dec(v_x_273_);
return v_res_281_;
}
}
static lean_object* _init_lp_mathlib_Simps_nsmulArgs___redArg___closed__2(void){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_285_ = lean_box(0);
v___x_286_ = ((lean_object*)(lp_mathlib_Simps_nsmulArgs___redArg___closed__1));
v___x_287_ = l_Lean_Expr_const___override(v___x_286_, v___x_285_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib_Simps_nsmulArgs___redArg___closed__3(void){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_288_ = lean_obj_once(&lp_mathlib_Simps_nsmulArgs___redArg___closed__2, &lp_mathlib_Simps_nsmulArgs___redArg___closed__2_once, _init_lp_mathlib_Simps_nsmulArgs___redArg___closed__2);
v___x_289_ = lean_unsigned_to_nat(2u);
v___x_290_ = lean_mk_empty_array_with_capacity(v___x_289_);
v___x_291_ = lean_array_push(v___x_290_, v___x_288_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_nsmulArgs___redArg(lean_object* v_args_292_){
_start:
{
lean_object* v___y_295_; lean_object* v___x_303_; lean_object* v___x_304_; uint8_t v___x_305_; 
v___x_303_ = lean_unsigned_to_nat(0u);
v___x_304_ = lean_array_get_size(v_args_292_);
v___x_305_ = lean_nat_dec_lt(v___x_303_, v___x_304_);
if (v___x_305_ == 0)
{
lean_object* v___x_306_; 
v___x_306_ = lean_obj_once(&lp_mathlib_Simps_copyFirst___redArg___closed__2, &lp_mathlib_Simps_copyFirst___redArg___closed__2_once, _init_lp_mathlib_Simps_copyFirst___redArg___closed__2);
v___y_295_ = v___x_306_;
goto v___jp_294_;
}
else
{
lean_object* v___x_307_; 
v___x_307_ = lean_array_fget_borrowed(v_args_292_, v___x_303_);
lean_inc(v___x_307_);
v___y_295_ = v___x_307_;
goto v___jp_294_;
}
v___jp_294_:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; size_t v_sz_299_; size_t v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; 
v___x_296_ = lean_obj_once(&lp_mathlib_Simps_nsmulArgs___redArg___closed__3, &lp_mathlib_Simps_nsmulArgs___redArg___closed__3_once, _init_lp_mathlib_Simps_nsmulArgs___redArg___closed__3);
v___x_297_ = lean_array_push(v___x_296_, v___y_295_);
v___x_298_ = l_Array_append___redArg(v___x_297_, v_args_292_);
v_sz_299_ = lean_array_size(v___x_298_);
v___x_300_ = ((size_t)0ULL);
v___x_301_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1(v_sz_299_, v___x_300_, v___x_298_);
v___x_302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_302_, 0, v___x_301_);
return v___x_302_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_nsmulArgs___redArg___boxed(lean_object* v_args_308_, lean_object* v_a_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_mathlib_Simps_nsmulArgs___redArg(v_args_308_);
lean_dec_ref(v_args_308_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_nsmulArgs(lean_object* v_x_311_, lean_object* v_x_312_, lean_object* v_args_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_, lean_object* v_a_317_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lp_mathlib_Simps_nsmulArgs___redArg(v_args_313_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_nsmulArgs___boxed(lean_object* v_x_320_, lean_object* v_x_321_, lean_object* v_args_322_, lean_object* v_a_323_, lean_object* v_a_324_, lean_object* v_a_325_, lean_object* v_a_326_, lean_object* v_a_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_Simps_nsmulArgs(v_x_320_, v_x_321_, v_args_322_, v_a_323_, v_a_324_, v_a_325_, v_a_326_);
lean_dec(v_a_326_);
lean_dec_ref(v_a_325_);
lean_dec(v_a_324_);
lean_dec_ref(v_a_323_);
lean_dec_ref(v_args_322_);
lean_dec(v_x_321_);
lean_dec(v_x_320_);
return v_res_328_;
}
}
static lean_object* _init_lp_mathlib_Simps_zsmulArgs___redArg___closed__2(void){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; 
v___x_332_ = lean_box(0);
v___x_333_ = ((lean_object*)(lp_mathlib_Simps_zsmulArgs___redArg___closed__1));
v___x_334_ = l_Lean_Expr_const___override(v___x_333_, v___x_332_);
return v___x_334_;
}
}
static lean_object* _init_lp_mathlib_Simps_zsmulArgs___redArg___closed__3(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_335_ = lean_obj_once(&lp_mathlib_Simps_zsmulArgs___redArg___closed__2, &lp_mathlib_Simps_zsmulArgs___redArg___closed__2_once, _init_lp_mathlib_Simps_zsmulArgs___redArg___closed__2);
v___x_336_ = lean_unsigned_to_nat(2u);
v___x_337_ = lean_mk_empty_array_with_capacity(v___x_336_);
v___x_338_ = lean_array_push(v___x_337_, v___x_335_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_zsmulArgs___redArg(lean_object* v_args_339_){
_start:
{
lean_object* v___y_342_; lean_object* v___x_350_; lean_object* v___x_351_; uint8_t v___x_352_; 
v___x_350_ = lean_unsigned_to_nat(0u);
v___x_351_ = lean_array_get_size(v_args_339_);
v___x_352_ = lean_nat_dec_lt(v___x_350_, v___x_351_);
if (v___x_352_ == 0)
{
lean_object* v___x_353_; 
v___x_353_ = lean_obj_once(&lp_mathlib_Simps_copyFirst___redArg___closed__2, &lp_mathlib_Simps_copyFirst___redArg___closed__2_once, _init_lp_mathlib_Simps_copyFirst___redArg___closed__2);
v___y_342_ = v___x_353_;
goto v___jp_341_;
}
else
{
lean_object* v___x_354_; 
v___x_354_ = lean_array_fget_borrowed(v_args_339_, v___x_350_);
lean_inc(v___x_354_);
v___y_342_ = v___x_354_;
goto v___jp_341_;
}
v___jp_341_:
{
lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; size_t v_sz_346_; size_t v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_343_ = lean_obj_once(&lp_mathlib_Simps_zsmulArgs___redArg___closed__3, &lp_mathlib_Simps_zsmulArgs___redArg___closed__3_once, _init_lp_mathlib_Simps_zsmulArgs___redArg___closed__3);
v___x_344_ = lean_array_push(v___x_343_, v___y_342_);
v___x_345_ = l_Array_append___redArg(v___x_344_, v_args_339_);
v_sz_346_ = lean_array_size(v___x_345_);
v___x_347_ = ((size_t)0ULL);
v___x_348_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Simps_defaultfindArgs_spec__1(v_sz_346_, v___x_347_, v___x_345_);
v___x_349_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_349_, 0, v___x_348_);
return v___x_349_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_zsmulArgs___redArg___boxed(lean_object* v_args_355_, lean_object* v_a_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_Simps_zsmulArgs___redArg(v_args_355_);
lean_dec_ref(v_args_355_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_zsmulArgs(lean_object* v_x_358_, lean_object* v_x_359_, lean_object* v_args_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lp_mathlib_Simps_zsmulArgs___redArg(v_args_360_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_zsmulArgs___boxed(lean_object* v_x_367_, lean_object* v_x_368_, lean_object* v_args_369_, lean_object* v_a_370_, lean_object* v_a_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_){
_start:
{
lean_object* v_res_375_; 
v_res_375_ = lp_mathlib_Simps_zsmulArgs(v_x_367_, v_x_368_, v_args_369_, v_a_370_, v_a_371_, v_a_372_, v_a_373_);
lean_dec(v_a_373_);
lean_dec_ref(v_a_372_);
lean_dec(v_a_371_);
lean_dec_ref(v_a_370_);
lean_dec_ref(v_args_369_);
lean_dec(v_x_368_);
lean_dec(v_x_367_);
return v_res_375_;
}
}
static lean_object* _init_lp_mathlib_Simps_findZeroArgs___redArg___closed__0(void){
_start:
{
lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_376_ = lean_unsigned_to_nat(0u);
v___x_377_ = l_Lean_mkRawNatLit(v___x_376_);
return v___x_377_;
}
}
static lean_object* _init_lp_mathlib_Simps_findZeroArgs___redArg___closed__1(void){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_378_ = lean_obj_once(&lp_mathlib_Simps_findZeroArgs___redArg___closed__0, &lp_mathlib_Simps_findZeroArgs___redArg___closed__0_once, _init_lp_mathlib_Simps_findZeroArgs___redArg___closed__0);
v___x_379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_379_, 0, v___x_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findZeroArgs___redArg(lean_object* v_args_380_){
_start:
{
lean_object* v___x_382_; lean_object* v___y_384_; lean_object* v___x_392_; uint8_t v___x_393_; 
v___x_382_ = lean_unsigned_to_nat(0u);
v___x_392_ = lean_array_get_size(v_args_380_);
v___x_393_ = lean_nat_dec_lt(v___x_382_, v___x_392_);
if (v___x_393_ == 0)
{
lean_object* v___x_394_; 
v___x_394_ = lean_obj_once(&lp_mathlib_Simps_copyFirst___redArg___closed__2, &lp_mathlib_Simps_copyFirst___redArg___closed__2_once, _init_lp_mathlib_Simps_copyFirst___redArg___closed__2);
v___y_384_ = v___x_394_;
goto v___jp_383_;
}
else
{
lean_object* v___x_395_; 
v___x_395_ = lean_array_fget_borrowed(v_args_380_, v___x_382_);
lean_inc(v___x_395_);
v___y_384_ = v___x_395_;
goto v___jp_383_;
}
v___jp_383_:
{
lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_385_, 0, v___y_384_);
v___x_386_ = lean_obj_once(&lp_mathlib_Simps_findZeroArgs___redArg___closed__1, &lp_mathlib_Simps_findZeroArgs___redArg___closed__1_once, _init_lp_mathlib_Simps_findZeroArgs___redArg___closed__1);
v___x_387_ = lean_unsigned_to_nat(2u);
v___x_388_ = lean_mk_empty_array_with_capacity(v___x_387_);
v___x_389_ = lean_array_push(v___x_388_, v___x_385_);
v___x_390_ = lean_array_push(v___x_389_, v___x_386_);
v___x_391_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_391_, 0, v___x_390_);
return v___x_391_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findZeroArgs___redArg___boxed(lean_object* v_args_396_, lean_object* v_a_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_mathlib_Simps_findZeroArgs___redArg(v_args_396_);
lean_dec_ref(v_args_396_);
return v_res_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findZeroArgs(lean_object* v_x_399_, lean_object* v_x_400_, lean_object* v_args_401_, lean_object* v_a_402_, lean_object* v_a_403_, lean_object* v_a_404_, lean_object* v_a_405_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lp_mathlib_Simps_findZeroArgs___redArg(v_args_401_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findZeroArgs___boxed(lean_object* v_x_408_, lean_object* v_x_409_, lean_object* v_args_410_, lean_object* v_a_411_, lean_object* v_a_412_, lean_object* v_a_413_, lean_object* v_a_414_, lean_object* v_a_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_Simps_findZeroArgs(v_x_408_, v_x_409_, v_args_410_, v_a_411_, v_a_412_, v_a_413_, v_a_414_);
lean_dec(v_a_414_);
lean_dec_ref(v_a_413_);
lean_dec(v_a_412_);
lean_dec_ref(v_a_411_);
lean_dec_ref(v_args_410_);
lean_dec(v_x_409_);
lean_dec(v_x_408_);
return v_res_416_;
}
}
static lean_object* _init_lp_mathlib_Simps_findOneArgs___redArg___closed__0(void){
_start:
{
lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_417_ = lean_unsigned_to_nat(1u);
v___x_418_ = l_Lean_mkRawNatLit(v___x_417_);
return v___x_418_;
}
}
static lean_object* _init_lp_mathlib_Simps_findOneArgs___redArg___closed__1(void){
_start:
{
lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_419_ = lean_obj_once(&lp_mathlib_Simps_findOneArgs___redArg___closed__0, &lp_mathlib_Simps_findOneArgs___redArg___closed__0_once, _init_lp_mathlib_Simps_findOneArgs___redArg___closed__0);
v___x_420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findOneArgs___redArg(lean_object* v_args_421_){
_start:
{
lean_object* v___y_424_; lean_object* v___x_432_; lean_object* v___x_433_; uint8_t v___x_434_; 
v___x_432_ = lean_unsigned_to_nat(0u);
v___x_433_ = lean_array_get_size(v_args_421_);
v___x_434_ = lean_nat_dec_lt(v___x_432_, v___x_433_);
if (v___x_434_ == 0)
{
lean_object* v___x_435_; 
v___x_435_ = lean_obj_once(&lp_mathlib_Simps_copyFirst___redArg___closed__2, &lp_mathlib_Simps_copyFirst___redArg___closed__2_once, _init_lp_mathlib_Simps_copyFirst___redArg___closed__2);
v___y_424_ = v___x_435_;
goto v___jp_423_;
}
else
{
lean_object* v___x_436_; 
v___x_436_ = lean_array_fget_borrowed(v_args_421_, v___x_432_);
lean_inc(v___x_436_);
v___y_424_ = v___x_436_;
goto v___jp_423_;
}
v___jp_423_:
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_425_, 0, v___y_424_);
v___x_426_ = lean_obj_once(&lp_mathlib_Simps_findOneArgs___redArg___closed__1, &lp_mathlib_Simps_findOneArgs___redArg___closed__1_once, _init_lp_mathlib_Simps_findOneArgs___redArg___closed__1);
v___x_427_ = lean_unsigned_to_nat(2u);
v___x_428_ = lean_mk_empty_array_with_capacity(v___x_427_);
v___x_429_ = lean_array_push(v___x_428_, v___x_425_);
v___x_430_ = lean_array_push(v___x_429_, v___x_426_);
v___x_431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_431_, 0, v___x_430_);
return v___x_431_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findOneArgs___redArg___boxed(lean_object* v_args_437_, lean_object* v_a_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_Simps_findOneArgs___redArg(v_args_437_);
lean_dec_ref(v_args_437_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findOneArgs(lean_object* v_x_440_, lean_object* v_x_441_, lean_object* v_args_442_, lean_object* v_a_443_, lean_object* v_a_444_, lean_object* v_a_445_, lean_object* v_a_446_){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lp_mathlib_Simps_findOneArgs___redArg(v_args_442_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findOneArgs___boxed(lean_object* v_x_449_, lean_object* v_x_450_, lean_object* v_args_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_mathlib_Simps_findOneArgs(v_x_449_, v_x_450_, v_args_451_, v_a_452_, v_a_453_, v_a_454_, v_a_455_);
lean_dec(v_a_455_);
lean_dec_ref(v_a_454_);
lean_dec(v_a_453_);
lean_dec_ref(v_a_452_);
lean_dec_ref(v_args_451_);
lean_dec(v_x_450_);
lean_dec(v_x_449_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__1(lean_object* v_a_458_, lean_object* v_a_459_){
_start:
{
if (lean_obj_tag(v_a_458_) == 0)
{
lean_object* v___x_460_; 
v___x_460_ = l_List_reverse___redArg(v_a_459_);
return v___x_460_;
}
else
{
lean_object* v_head_461_; lean_object* v_tail_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_471_; 
v_head_461_ = lean_ctor_get(v_a_458_, 0);
v_tail_462_ = lean_ctor_get(v_a_458_, 1);
v_isSharedCheck_471_ = !lean_is_exclusive(v_a_458_);
if (v_isSharedCheck_471_ == 0)
{
v___x_464_ = v_a_458_;
v_isShared_465_ = v_isSharedCheck_471_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_tail_462_);
lean_inc(v_head_461_);
lean_dec(v_a_458_);
v___x_464_ = lean_box(0);
v_isShared_465_ = v_isSharedCheck_471_;
goto v_resetjp_463_;
}
v_resetjp_463_:
{
lean_object* v___x_466_; lean_object* v___x_468_; 
v___x_466_ = l_Lean_mkLevelParam(v_head_461_);
if (v_isShared_465_ == 0)
{
lean_ctor_set(v___x_464_, 1, v_a_459_);
lean_ctor_set(v___x_464_, 0, v___x_466_);
v___x_468_ = v___x_464_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v___x_466_);
lean_ctor_set(v_reuseFailAlloc_470_, 1, v_a_459_);
v___x_468_ = v_reuseFailAlloc_470_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
v_a_458_ = v_tail_462_;
v_a_459_ = v___x_468_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(lean_object* v_ref_472_, lean_object* v_msg_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_){
_start:
{
lean_object* v_fileName_479_; lean_object* v_fileMap_480_; lean_object* v_options_481_; lean_object* v_currRecDepth_482_; lean_object* v_maxRecDepth_483_; lean_object* v_ref_484_; lean_object* v_currNamespace_485_; lean_object* v_openDecls_486_; lean_object* v_initHeartbeats_487_; lean_object* v_maxHeartbeats_488_; lean_object* v_quotContext_489_; lean_object* v_currMacroScope_490_; uint8_t v_diag_491_; lean_object* v_cancelTk_x3f_492_; uint8_t v_suppressElabErrors_493_; lean_object* v_inheritedTraceOptions_494_; lean_object* v_ref_495_; lean_object* v___x_496_; lean_object* v___x_497_; 
v_fileName_479_ = lean_ctor_get(v___y_476_, 0);
v_fileMap_480_ = lean_ctor_get(v___y_476_, 1);
v_options_481_ = lean_ctor_get(v___y_476_, 2);
v_currRecDepth_482_ = lean_ctor_get(v___y_476_, 3);
v_maxRecDepth_483_ = lean_ctor_get(v___y_476_, 4);
v_ref_484_ = lean_ctor_get(v___y_476_, 5);
v_currNamespace_485_ = lean_ctor_get(v___y_476_, 6);
v_openDecls_486_ = lean_ctor_get(v___y_476_, 7);
v_initHeartbeats_487_ = lean_ctor_get(v___y_476_, 8);
v_maxHeartbeats_488_ = lean_ctor_get(v___y_476_, 9);
v_quotContext_489_ = lean_ctor_get(v___y_476_, 10);
v_currMacroScope_490_ = lean_ctor_get(v___y_476_, 11);
v_diag_491_ = lean_ctor_get_uint8(v___y_476_, sizeof(void*)*14);
v_cancelTk_x3f_492_ = lean_ctor_get(v___y_476_, 12);
v_suppressElabErrors_493_ = lean_ctor_get_uint8(v___y_476_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_494_ = lean_ctor_get(v___y_476_, 13);
v_ref_495_ = l_Lean_replaceRef(v_ref_472_, v_ref_484_);
lean_inc_ref(v_inheritedTraceOptions_494_);
lean_inc(v_cancelTk_x3f_492_);
lean_inc(v_currMacroScope_490_);
lean_inc(v_quotContext_489_);
lean_inc(v_maxHeartbeats_488_);
lean_inc(v_initHeartbeats_487_);
lean_inc(v_openDecls_486_);
lean_inc(v_currNamespace_485_);
lean_inc(v_maxRecDepth_483_);
lean_inc(v_currRecDepth_482_);
lean_inc_ref(v_options_481_);
lean_inc_ref(v_fileMap_480_);
lean_inc_ref(v_fileName_479_);
v___x_496_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_496_, 0, v_fileName_479_);
lean_ctor_set(v___x_496_, 1, v_fileMap_480_);
lean_ctor_set(v___x_496_, 2, v_options_481_);
lean_ctor_set(v___x_496_, 3, v_currRecDepth_482_);
lean_ctor_set(v___x_496_, 4, v_maxRecDepth_483_);
lean_ctor_set(v___x_496_, 5, v_ref_495_);
lean_ctor_set(v___x_496_, 6, v_currNamespace_485_);
lean_ctor_set(v___x_496_, 7, v_openDecls_486_);
lean_ctor_set(v___x_496_, 8, v_initHeartbeats_487_);
lean_ctor_set(v___x_496_, 9, v_maxHeartbeats_488_);
lean_ctor_set(v___x_496_, 10, v_quotContext_489_);
lean_ctor_set(v___x_496_, 11, v_currMacroScope_490_);
lean_ctor_set(v___x_496_, 12, v_cancelTk_x3f_492_);
lean_ctor_set(v___x_496_, 13, v_inheritedTraceOptions_494_);
lean_ctor_set_uint8(v___x_496_, sizeof(void*)*14, v_diag_491_);
lean_ctor_set_uint8(v___x_496_, sizeof(void*)*14 + 1, v_suppressElabErrors_493_);
v___x_497_ = lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg(v_msg_473_, v___y_474_, v___y_475_, v___x_496_, v___y_477_);
lean_dec_ref_known(v___x_496_, 14);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg___boxed(lean_object* v_ref_498_, lean_object* v_msg_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_, lean_object* v___y_503_, lean_object* v___y_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(v_ref_498_, v_msg_499_, v___y_500_, v___y_501_, v___y_502_, v___y_503_);
lean_dec(v___y_503_);
lean_dec_ref(v___y_502_);
lean_dec(v___y_501_);
lean_dec_ref(v___y_500_);
lean_dec(v_ref_498_);
return v_res_505_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_506_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1(void){
_start:
{
lean_object* v___x_507_; lean_object* v___x_508_; 
v___x_507_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__0);
v___x_508_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_508_, 0, v___x_507_);
return v___x_508_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; 
v___x_509_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1);
v___x_510_ = lean_unsigned_to_nat(0u);
v___x_511_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_511_, 0, v___x_510_);
lean_ctor_set(v___x_511_, 1, v___x_510_);
lean_ctor_set(v___x_511_, 2, v___x_510_);
lean_ctor_set(v___x_511_, 3, v___x_510_);
lean_ctor_set(v___x_511_, 4, v___x_509_);
lean_ctor_set(v___x_511_, 5, v___x_509_);
lean_ctor_set(v___x_511_, 6, v___x_509_);
lean_ctor_set(v___x_511_, 7, v___x_509_);
lean_ctor_set(v___x_511_, 8, v___x_509_);
lean_ctor_set(v___x_511_, 9, v___x_509_);
return v___x_511_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3(void){
_start:
{
lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; 
v___x_512_ = lean_unsigned_to_nat(32u);
v___x_513_ = lean_mk_empty_array_with_capacity(v___x_512_);
v___x_514_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_514_, 0, v___x_513_);
return v___x_514_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4(void){
_start:
{
size_t v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; 
v___x_515_ = ((size_t)5ULL);
v___x_516_ = lean_unsigned_to_nat(0u);
v___x_517_ = lean_unsigned_to_nat(32u);
v___x_518_ = lean_mk_empty_array_with_capacity(v___x_517_);
v___x_519_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3);
v___x_520_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_520_, 0, v___x_519_);
lean_ctor_set(v___x_520_, 1, v___x_518_);
lean_ctor_set(v___x_520_, 2, v___x_516_);
lean_ctor_set(v___x_520_, 3, v___x_516_);
lean_ctor_set_usize(v___x_520_, 4, v___x_515_);
return v___x_520_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5(void){
_start:
{
lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; 
v___x_521_ = lean_box(1);
v___x_522_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__4);
v___x_523_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__1);
v___x_524_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_524_, 0, v___x_523_);
lean_ctor_set(v___x_524_, 1, v___x_522_);
lean_ctor_set(v___x_524_, 2, v___x_521_);
return v___x_524_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7(void){
_start:
{
lean_object* v___x_526_; lean_object* v___x_527_; 
v___x_526_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__6));
v___x_527_ = l_Lean_stringToMessageData(v___x_526_);
return v___x_527_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9(void){
_start:
{
lean_object* v___x_529_; lean_object* v___x_530_; 
v___x_529_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__8));
v___x_530_ = l_Lean_stringToMessageData(v___x_529_);
return v___x_530_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11(void){
_start:
{
lean_object* v___x_532_; lean_object* v___x_533_; 
v___x_532_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__10));
v___x_533_ = l_Lean_stringToMessageData(v___x_532_);
return v___x_533_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13(void){
_start:
{
lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_535_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__12));
v___x_536_ = l_Lean_stringToMessageData(v___x_535_);
return v___x_536_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__15(void){
_start:
{
lean_object* v___x_538_; lean_object* v___x_539_; 
v___x_538_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__14));
v___x_539_ = l_Lean_stringToMessageData(v___x_538_);
return v___x_539_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__17(void){
_start:
{
lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_541_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__16));
v___x_542_ = l_Lean_stringToMessageData(v___x_541_);
return v___x_542_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__19(void){
_start:
{
lean_object* v___x_544_; lean_object* v___x_545_; 
v___x_544_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__18));
v___x_545_ = l_Lean_stringToMessageData(v___x_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(lean_object* v_msg_546_, lean_object* v_declHint_547_, lean_object* v___y_548_){
_start:
{
lean_object* v___x_550_; lean_object* v_env_551_; uint8_t v___x_552_; 
v___x_550_ = lean_st_ref_get(v___y_548_);
v_env_551_ = lean_ctor_get(v___x_550_, 0);
lean_inc_ref(v_env_551_);
lean_dec(v___x_550_);
v___x_552_ = l_Lean_Name_isAnonymous(v_declHint_547_);
if (v___x_552_ == 0)
{
uint8_t v_isExporting_553_; 
v_isExporting_553_ = lean_ctor_get_uint8(v_env_551_, sizeof(void*)*8);
if (v_isExporting_553_ == 0)
{
lean_object* v___x_554_; 
lean_dec_ref(v_env_551_);
lean_dec(v_declHint_547_);
v___x_554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_554_, 0, v_msg_546_);
return v___x_554_;
}
else
{
lean_object* v___x_555_; uint8_t v___x_556_; 
lean_inc_ref(v_env_551_);
v___x_555_ = l_Lean_Environment_setExporting(v_env_551_, v___x_552_);
lean_inc(v_declHint_547_);
lean_inc_ref(v___x_555_);
v___x_556_ = l_Lean_Environment_contains(v___x_555_, v_declHint_547_, v_isExporting_553_);
if (v___x_556_ == 0)
{
lean_object* v___x_557_; 
lean_dec_ref(v___x_555_);
lean_dec_ref(v_env_551_);
lean_dec(v_declHint_547_);
v___x_557_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_557_, 0, v_msg_546_);
return v___x_557_;
}
else
{
lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v_c_563_; lean_object* v___x_564_; 
v___x_558_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2);
v___x_559_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5);
v___x_560_ = l_Lean_Options_empty;
v___x_561_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_561_, 0, v___x_555_);
lean_ctor_set(v___x_561_, 1, v___x_558_);
lean_ctor_set(v___x_561_, 2, v___x_559_);
lean_ctor_set(v___x_561_, 3, v___x_560_);
lean_inc(v_declHint_547_);
v___x_562_ = l_Lean_MessageData_ofConstName(v_declHint_547_, v___x_552_);
v_c_563_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_563_, 0, v___x_561_);
lean_ctor_set(v_c_563_, 1, v___x_562_);
v___x_564_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_551_, v_declHint_547_);
if (lean_obj_tag(v___x_564_) == 0)
{
lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; 
lean_dec_ref(v_env_551_);
lean_dec(v_declHint_547_);
v___x_565_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7);
v___x_566_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_566_, 0, v___x_565_);
lean_ctor_set(v___x_566_, 1, v_c_563_);
v___x_567_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__9);
v___x_568_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_568_, 0, v___x_566_);
lean_ctor_set(v___x_568_, 1, v___x_567_);
v___x_569_ = l_Lean_MessageData_note(v___x_568_);
v___x_570_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_570_, 0, v_msg_546_);
lean_ctor_set(v___x_570_, 1, v___x_569_);
v___x_571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_571_, 0, v___x_570_);
return v___x_571_;
}
else
{
lean_object* v_val_572_; lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_607_; 
v_val_572_ = lean_ctor_get(v___x_564_, 0);
v_isSharedCheck_607_ = !lean_is_exclusive(v___x_564_);
if (v_isSharedCheck_607_ == 0)
{
v___x_574_ = v___x_564_;
v_isShared_575_ = v_isSharedCheck_607_;
goto v_resetjp_573_;
}
else
{
lean_inc(v_val_572_);
lean_dec(v___x_564_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_607_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v_mod_579_; uint8_t v___x_580_; 
v___x_576_ = lean_box(0);
v___x_577_ = l_Lean_Environment_header(v_env_551_);
lean_dec_ref(v_env_551_);
v___x_578_ = l_Lean_EnvironmentHeader_moduleNames(v___x_577_);
v_mod_579_ = lean_array_get(v___x_576_, v___x_578_, v_val_572_);
lean_dec(v_val_572_);
lean_dec_ref(v___x_578_);
v___x_580_ = l_Lean_isPrivateName(v_declHint_547_);
lean_dec(v_declHint_547_);
if (v___x_580_ == 0)
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_592_; 
v___x_581_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__11);
v___x_582_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_582_, 0, v___x_581_);
lean_ctor_set(v___x_582_, 1, v_c_563_);
v___x_583_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__13);
v___x_584_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_584_, 0, v___x_582_);
lean_ctor_set(v___x_584_, 1, v___x_583_);
v___x_585_ = l_Lean_MessageData_ofName(v_mod_579_);
v___x_586_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_586_, 0, v___x_584_);
lean_ctor_set(v___x_586_, 1, v___x_585_);
v___x_587_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__15, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__15_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__15);
v___x_588_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_588_, 0, v___x_586_);
lean_ctor_set(v___x_588_, 1, v___x_587_);
v___x_589_ = l_Lean_MessageData_note(v___x_588_);
v___x_590_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_590_, 0, v_msg_546_);
lean_ctor_set(v___x_590_, 1, v___x_589_);
if (v_isShared_575_ == 0)
{
lean_ctor_set_tag(v___x_574_, 0);
lean_ctor_set(v___x_574_, 0, v___x_590_);
v___x_592_ = v___x_574_;
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
lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_605_; 
v___x_594_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__7);
v___x_595_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_595_, 0, v___x_594_);
lean_ctor_set(v___x_595_, 1, v_c_563_);
v___x_596_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__17, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__17_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__17);
v___x_597_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_597_, 0, v___x_595_);
lean_ctor_set(v___x_597_, 1, v___x_596_);
v___x_598_ = l_Lean_MessageData_ofName(v_mod_579_);
v___x_599_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_599_, 0, v___x_597_);
lean_ctor_set(v___x_599_, 1, v___x_598_);
v___x_600_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__19, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__19_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__19);
v___x_601_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_601_, 0, v___x_599_);
lean_ctor_set(v___x_601_, 1, v___x_600_);
v___x_602_ = l_Lean_MessageData_note(v___x_601_);
v___x_603_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_603_, 0, v_msg_546_);
lean_ctor_set(v___x_603_, 1, v___x_602_);
if (v_isShared_575_ == 0)
{
lean_ctor_set_tag(v___x_574_, 0);
lean_ctor_set(v___x_574_, 0, v___x_603_);
v___x_605_ = v___x_574_;
goto v_reusejp_604_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v___x_603_);
v___x_605_ = v_reuseFailAlloc_606_;
goto v_reusejp_604_;
}
v_reusejp_604_:
{
return v___x_605_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_608_; 
lean_dec_ref(v_env_551_);
lean_dec(v_declHint_547_);
v___x_608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_608_, 0, v_msg_546_);
return v___x_608_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___boxed(lean_object* v_msg_609_, lean_object* v_declHint_610_, lean_object* v___y_611_, lean_object* v___y_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(v_msg_609_, v_declHint_610_, v___y_611_);
lean_dec(v___y_611_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5(lean_object* v_msg_614_, lean_object* v_declHint_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_){
_start:
{
lean_object* v___x_621_; lean_object* v_a_622_; lean_object* v___x_624_; uint8_t v_isShared_625_; uint8_t v_isSharedCheck_631_; 
v___x_621_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(v_msg_614_, v_declHint_615_, v___y_619_);
v_a_622_ = lean_ctor_get(v___x_621_, 0);
v_isSharedCheck_631_ = !lean_is_exclusive(v___x_621_);
if (v_isSharedCheck_631_ == 0)
{
v___x_624_ = v___x_621_;
v_isShared_625_ = v_isSharedCheck_631_;
goto v_resetjp_623_;
}
else
{
lean_inc(v_a_622_);
lean_dec(v___x_621_);
v___x_624_ = lean_box(0);
v_isShared_625_ = v_isSharedCheck_631_;
goto v_resetjp_623_;
}
v_resetjp_623_:
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_629_; 
v___x_626_ = l_Lean_unknownIdentifierMessageTag;
v___x_627_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_627_, 0, v___x_626_);
lean_ctor_set(v___x_627_, 1, v_a_622_);
if (v_isShared_625_ == 0)
{
lean_ctor_set(v___x_624_, 0, v___x_627_);
v___x_629_ = v___x_624_;
goto v_reusejp_628_;
}
else
{
lean_object* v_reuseFailAlloc_630_; 
v_reuseFailAlloc_630_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_630_, 0, v___x_627_);
v___x_629_ = v_reuseFailAlloc_630_;
goto v_reusejp_628_;
}
v_reusejp_628_:
{
return v___x_629_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5___boxed(lean_object* v_msg_632_, lean_object* v_declHint_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5(v_msg_632_, v_declHint_633_, v___y_634_, v___y_635_, v___y_636_, v___y_637_);
lean_dec(v___y_637_);
lean_dec_ref(v___y_636_);
lean_dec(v___y_635_);
lean_dec_ref(v___y_634_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_ref_640_, lean_object* v_msg_641_, lean_object* v_declHint_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_){
_start:
{
lean_object* v___x_648_; lean_object* v_a_649_; lean_object* v___x_650_; 
v___x_648_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5(v_msg_641_, v_declHint_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
v_a_649_ = lean_ctor_get(v___x_648_, 0);
lean_inc(v_a_649_);
lean_dec_ref(v___x_648_);
v___x_650_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(v_ref_640_, v_a_649_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_ref_651_, lean_object* v_msg_652_, lean_object* v_declHint_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_){
_start:
{
lean_object* v_res_659_; 
v_res_659_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_ref_651_, v_msg_652_, v_declHint_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
lean_dec(v_ref_651_);
return v_res_659_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_661_; lean_object* v___x_662_; 
v___x_661_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__0));
v___x_662_ = l_Lean_stringToMessageData(v___x_661_);
return v___x_662_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_664_; lean_object* v___x_665_; 
v___x_664_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__2));
v___x_665_ = l_Lean_stringToMessageData(v___x_664_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_ref_666_, lean_object* v_constName_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_){
_start:
{
lean_object* v___x_673_; uint8_t v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; 
v___x_673_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__1);
v___x_674_ = 0;
lean_inc(v_constName_667_);
v___x_675_ = l_Lean_MessageData_ofConstName(v_constName_667_, v___x_674_);
v___x_676_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_676_, 0, v___x_673_);
lean_ctor_set(v___x_676_, 1, v___x_675_);
v___x_677_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__3, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__3_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___closed__3);
v___x_678_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_678_, 0, v___x_676_);
lean_ctor_set(v___x_678_, 1, v___x_677_);
v___x_679_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_ref_666_, v___x_678_, v_constName_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_ref_680_, lean_object* v_constName_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_, lean_object* v___y_686_){
_start:
{
lean_object* v_res_687_; 
v_res_687_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg(v_ref_680_, v_constName_681_, v___y_682_, v___y_683_, v___y_684_, v___y_685_);
lean_dec(v___y_685_);
lean_dec_ref(v___y_684_);
lean_dec(v___y_683_);
lean_dec_ref(v___y_682_);
lean_dec(v_ref_680_);
return v_res_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___redArg(lean_object* v_constName_688_, lean_object* v___y_689_, lean_object* v___y_690_, lean_object* v___y_691_, lean_object* v___y_692_){
_start:
{
lean_object* v_ref_694_; lean_object* v___x_695_; 
v_ref_694_ = lean_ctor_get(v___y_691_, 5);
v___x_695_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg(v_ref_694_, v_constName_688_, v___y_689_, v___y_690_, v___y_691_, v___y_692_);
return v___x_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_constName_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___redArg(v_constName_696_, v___y_697_, v___y_698_, v___y_699_, v___y_700_);
lean_dec(v___y_700_);
lean_dec_ref(v___y_699_);
lean_dec(v___y_698_);
lean_dec_ref(v___y_697_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0(lean_object* v_constName_703_, lean_object* v___y_704_, lean_object* v___y_705_, lean_object* v___y_706_, lean_object* v___y_707_){
_start:
{
lean_object* v___x_709_; lean_object* v_env_710_; uint8_t v___x_711_; lean_object* v___x_712_; 
v___x_709_ = lean_st_ref_get(v___y_707_);
v_env_710_ = lean_ctor_get(v___x_709_, 0);
lean_inc_ref(v_env_710_);
lean_dec(v___x_709_);
v___x_711_ = 0;
lean_inc(v_constName_703_);
v___x_712_ = l_Lean_Environment_findConstVal_x3f(v_env_710_, v_constName_703_, v___x_711_);
if (lean_obj_tag(v___x_712_) == 0)
{
lean_object* v___x_713_; 
v___x_713_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___redArg(v_constName_703_, v___y_704_, v___y_705_, v___y_706_, v___y_707_);
return v___x_713_;
}
else
{
lean_object* v_val_714_; lean_object* v___x_716_; uint8_t v_isShared_717_; uint8_t v_isSharedCheck_721_; 
lean_dec(v_constName_703_);
v_val_714_ = lean_ctor_get(v___x_712_, 0);
v_isSharedCheck_721_ = !lean_is_exclusive(v___x_712_);
if (v_isSharedCheck_721_ == 0)
{
v___x_716_ = v___x_712_;
v_isShared_717_ = v_isSharedCheck_721_;
goto v_resetjp_715_;
}
else
{
lean_inc(v_val_714_);
lean_dec(v___x_712_);
v___x_716_ = lean_box(0);
v_isShared_717_ = v_isSharedCheck_721_;
goto v_resetjp_715_;
}
v_resetjp_715_:
{
lean_object* v___x_719_; 
if (v_isShared_717_ == 0)
{
lean_ctor_set_tag(v___x_716_, 0);
v___x_719_ = v___x_716_;
goto v_reusejp_718_;
}
else
{
lean_object* v_reuseFailAlloc_720_; 
v_reuseFailAlloc_720_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_720_, 0, v_val_714_);
v___x_719_ = v_reuseFailAlloc_720_;
goto v_reusejp_718_;
}
v_reusejp_718_:
{
return v___x_719_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0___boxed(lean_object* v_constName_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_){
_start:
{
lean_object* v_res_728_; 
v_res_728_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0(v_constName_722_, v___y_723_, v___y_724_, v___y_725_, v___y_726_);
lean_dec(v___y_726_);
lean_dec_ref(v___y_725_);
lean_dec(v___y_724_);
lean_dec_ref(v___y_723_);
return v_res_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0(lean_object* v_constName_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_){
_start:
{
lean_object* v___x_735_; 
lean_inc(v_constName_729_);
v___x_735_ = lp_mathlib_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0(v_constName_729_, v___y_730_, v___y_731_, v___y_732_, v___y_733_);
if (lean_obj_tag(v___x_735_) == 0)
{
lean_object* v_a_736_; lean_object* v___x_738_; uint8_t v_isShared_739_; uint8_t v_isSharedCheck_747_; 
v_a_736_ = lean_ctor_get(v___x_735_, 0);
v_isSharedCheck_747_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_747_ == 0)
{
v___x_738_ = v___x_735_;
v_isShared_739_ = v_isSharedCheck_747_;
goto v_resetjp_737_;
}
else
{
lean_inc(v_a_736_);
lean_dec(v___x_735_);
v___x_738_ = lean_box(0);
v_isShared_739_ = v_isSharedCheck_747_;
goto v_resetjp_737_;
}
v_resetjp_737_:
{
lean_object* v_levelParams_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_745_; 
v_levelParams_740_ = lean_ctor_get(v_a_736_, 1);
lean_inc(v_levelParams_740_);
lean_dec(v_a_736_);
v___x_741_ = lean_box(0);
v___x_742_ = lp_mathlib_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__1(v_levelParams_740_, v___x_741_);
v___x_743_ = l_Lean_mkConst(v_constName_729_, v___x_742_);
if (v_isShared_739_ == 0)
{
lean_ctor_set(v___x_738_, 0, v___x_743_);
v___x_745_ = v___x_738_;
goto v_reusejp_744_;
}
else
{
lean_object* v_reuseFailAlloc_746_; 
v_reuseFailAlloc_746_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_746_, 0, v___x_743_);
v___x_745_ = v_reuseFailAlloc_746_;
goto v_reusejp_744_;
}
v_reusejp_744_:
{
return v___x_745_;
}
}
}
else
{
lean_object* v_a_748_; lean_object* v___x_750_; uint8_t v_isShared_751_; uint8_t v_isSharedCheck_755_; 
lean_dec(v_constName_729_);
v_a_748_ = lean_ctor_get(v___x_735_, 0);
v_isSharedCheck_755_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_755_ == 0)
{
v___x_750_ = v___x_735_;
v_isShared_751_ = v_isSharedCheck_755_;
goto v_resetjp_749_;
}
else
{
lean_inc(v_a_748_);
lean_dec(v___x_735_);
v___x_750_ = lean_box(0);
v_isShared_751_ = v_isSharedCheck_755_;
goto v_resetjp_749_;
}
v_resetjp_749_:
{
lean_object* v___x_753_; 
if (v_isShared_751_ == 0)
{
v___x_753_ = v___x_750_;
goto v_reusejp_752_;
}
else
{
lean_object* v_reuseFailAlloc_754_; 
v_reuseFailAlloc_754_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_754_, 0, v_a_748_);
v___x_753_ = v_reuseFailAlloc_754_;
goto v_reusejp_752_;
}
v_reusejp_752_:
{
return v___x_753_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0___boxed(lean_object* v_constName_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_){
_start:
{
lean_object* v_res_762_; 
v_res_762_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0(v_constName_756_, v___y_757_, v___y_758_, v___y_759_, v___y_760_);
lean_dec(v___y_760_);
lean_dec_ref(v___y_759_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_757_);
return v_res_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findCoercionArgs(lean_object* v_str_763_, lean_object* v_className_764_, lean_object* v_args_765_, lean_object* v_a_766_, lean_object* v_a_767_, lean_object* v_a_768_, lean_object* v_a_769_){
_start:
{
lean_object* v___x_771_; lean_object* v_env_772_; uint8_t v___x_773_; lean_object* v___x_774_; 
v___x_771_ = lean_st_ref_get(v_a_769_);
v_env_772_ = lean_ctor_get(v___x_771_, 0);
lean_inc_ref(v_env_772_);
lean_dec(v___x_771_);
v___x_773_ = 0;
lean_inc(v_className_764_);
v___x_774_ = l_Lean_Environment_find_x3f(v_env_772_, v_className_764_, v___x_773_);
if (lean_obj_tag(v___x_774_) == 1)
{
lean_object* v_val_775_; lean_object* v___x_777_; uint8_t v_isShared_778_; uint8_t v_isSharedCheck_809_; 
lean_dec(v_className_764_);
v_val_775_ = lean_ctor_get(v___x_774_, 0);
v_isSharedCheck_809_ = !lean_is_exclusive(v___x_774_);
if (v_isSharedCheck_809_ == 0)
{
v___x_777_ = v___x_774_;
v_isShared_778_ = v_isSharedCheck_809_;
goto v_resetjp_776_;
}
else
{
lean_inc(v_val_775_);
lean_dec(v___x_774_);
v___x_777_ = lean_box(0);
v_isShared_778_ = v_isSharedCheck_809_;
goto v_resetjp_776_;
}
v_resetjp_776_:
{
lean_object* v___x_779_; 
v___x_779_ = lp_mathlib_Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0(v_str_763_, v_a_766_, v_a_767_, v_a_768_, v_a_769_);
if (lean_obj_tag(v___x_779_) == 0)
{
lean_object* v_a_780_; lean_object* v___x_782_; uint8_t v_isShared_783_; uint8_t v_isSharedCheck_800_; 
v_a_780_ = lean_ctor_get(v___x_779_, 0);
v_isSharedCheck_800_ = !lean_is_exclusive(v___x_779_);
if (v_isSharedCheck_800_ == 0)
{
v___x_782_ = v___x_779_;
v_isShared_783_ = v_isSharedCheck_800_;
goto v_resetjp_781_;
}
else
{
lean_inc(v_a_780_);
lean_dec(v___x_779_);
v___x_782_ = lean_box(0);
v_isShared_783_ = v_isSharedCheck_800_;
goto v_resetjp_781_;
}
v_resetjp_781_:
{
lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_792_; 
v___x_784_ = l_Lean_ConstantInfo_type(v_val_775_);
lean_dec(v_val_775_);
v___x_785_ = l_Lean_Expr_getNumHeadForalls(v___x_784_);
lean_dec_ref(v___x_784_);
v___x_786_ = l_Lean_mkAppN(v_a_780_, v_args_765_);
v___x_787_ = lean_unsigned_to_nat(1u);
v___x_788_ = lean_nat_sub(v___x_785_, v___x_787_);
lean_dec(v___x_785_);
v___x_789_ = lean_box(0);
v___x_790_ = lean_mk_array(v___x_788_, v___x_789_);
if (v_isShared_778_ == 0)
{
lean_ctor_set(v___x_777_, 0, v___x_786_);
v___x_792_ = v___x_777_;
goto v_reusejp_791_;
}
else
{
lean_object* v_reuseFailAlloc_799_; 
v_reuseFailAlloc_799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_799_, 0, v___x_786_);
v___x_792_ = v_reuseFailAlloc_799_;
goto v_reusejp_791_;
}
v_reusejp_791_:
{
lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_797_; 
v___x_793_ = lean_mk_empty_array_with_capacity(v___x_787_);
v___x_794_ = lean_array_push(v___x_793_, v___x_792_);
v___x_795_ = l_Array_append___redArg(v___x_794_, v___x_790_);
lean_dec_ref(v___x_790_);
if (v_isShared_783_ == 0)
{
lean_ctor_set(v___x_782_, 0, v___x_795_);
v___x_797_ = v___x_782_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v___x_795_);
v___x_797_ = v_reuseFailAlloc_798_;
goto v_reusejp_796_;
}
v_reusejp_796_:
{
return v___x_797_;
}
}
}
}
else
{
lean_object* v_a_801_; lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_808_; 
lean_del_object(v___x_777_);
lean_dec(v_val_775_);
v_a_801_ = lean_ctor_get(v___x_779_, 0);
v_isSharedCheck_808_ = !lean_is_exclusive(v___x_779_);
if (v_isSharedCheck_808_ == 0)
{
v___x_803_ = v___x_779_;
v_isShared_804_ = v_isSharedCheck_808_;
goto v_resetjp_802_;
}
else
{
lean_inc(v_a_801_);
lean_dec(v___x_779_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_808_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v___x_806_; 
if (v_isShared_804_ == 0)
{
v___x_806_ = v___x_803_;
goto v_reusejp_805_;
}
else
{
lean_object* v_reuseFailAlloc_807_; 
v_reuseFailAlloc_807_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_807_, 0, v_a_801_);
v___x_806_ = v_reuseFailAlloc_807_;
goto v_reusejp_805_;
}
v_reusejp_805_:
{
return v___x_806_;
}
}
}
}
}
else
{
lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
lean_dec(v___x_774_);
lean_dec(v_str_763_);
v___x_810_ = lean_obj_once(&lp_mathlib_Simps_defaultfindArgs___redArg___closed__3, &lp_mathlib_Simps_defaultfindArgs___redArg___closed__3_once, _init_lp_mathlib_Simps_defaultfindArgs___redArg___closed__3);
v___x_811_ = l_Lean_MessageData_ofName(v_className_764_);
v___x_812_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_812_, 0, v___x_810_);
lean_ctor_set(v___x_812_, 1, v___x_811_);
v___x_813_ = lp_mathlib_Lean_throwError___at___00Simps_defaultfindArgs_spec__0___redArg(v___x_812_, v_a_766_, v_a_767_, v_a_768_, v_a_769_);
return v___x_813_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Simps_findCoercionArgs___boxed(lean_object* v_str_814_, lean_object* v_className_815_, lean_object* v_args_816_, lean_object* v_a_817_, lean_object* v_a_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_mathlib_Simps_findCoercionArgs(v_str_814_, v_className_815_, v_args_816_, v_a_817_, v_a_818_, v_a_819_, v_a_820_);
lean_dec(v_a_820_);
lean_dec_ref(v_a_819_);
lean_dec(v_a_818_);
lean_dec_ref(v_a_817_);
lean_dec_ref(v_args_816_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_823_, lean_object* v_constName_824_, lean_object* v___y_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_){
_start:
{
lean_object* v___x_830_; 
v___x_830_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___redArg(v_constName_824_, v___y_825_, v___y_826_, v___y_827_, v___y_828_);
return v___x_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_831_, lean_object* v_constName_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_){
_start:
{
lean_object* v_res_838_; 
v_res_838_ = lp_mathlib_Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1(v_00_u03b1_831_, v_constName_832_, v___y_833_, v___y_834_, v___y_835_, v___y_836_);
lean_dec(v___y_836_);
lean_dec_ref(v___y_835_);
lean_dec(v___y_834_);
lean_dec_ref(v___y_833_);
return v_res_838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b1_839_, lean_object* v_ref_840_, lean_object* v_constName_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_){
_start:
{
lean_object* v___x_847_; 
v___x_847_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___redArg(v_ref_840_, v_constName_841_, v___y_842_, v___y_843_, v___y_844_, v___y_845_);
return v___x_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b1_848_, lean_object* v_ref_849_, lean_object* v_constName_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_){
_start:
{
lean_object* v_res_856_; 
v_res_856_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2(v_00_u03b1_848_, v_ref_849_, v_constName_850_, v___y_851_, v___y_852_, v___y_853_, v___y_854_);
lean_dec(v___y_854_);
lean_dec_ref(v___y_853_);
lean_dec(v___y_852_);
lean_dec_ref(v___y_851_);
lean_dec(v_ref_849_);
return v_res_856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b1_857_, lean_object* v_ref_858_, lean_object* v_msg_859_, lean_object* v_declHint_860_, lean_object* v___y_861_, lean_object* v___y_862_, lean_object* v___y_863_, lean_object* v___y_864_){
_start:
{
lean_object* v___x_866_; 
v___x_866_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___redArg(v_ref_858_, v_msg_859_, v_declHint_860_, v___y_861_, v___y_862_, v___y_863_, v___y_864_);
return v___x_866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03b1_867_, lean_object* v_ref_868_, lean_object* v_msg_869_, lean_object* v_declHint_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_){
_start:
{
lean_object* v_res_876_; 
v_res_876_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4(v_00_u03b1_867_, v_ref_868_, v_msg_869_, v_declHint_870_, v___y_871_, v___y_872_, v___y_873_, v___y_874_);
lean_dec(v___y_874_);
lean_dec_ref(v___y_873_);
lean_dec(v___y_872_);
lean_dec_ref(v___y_871_);
lean_dec(v_ref_868_);
return v_res_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6(lean_object* v_msg_877_, lean_object* v_declHint_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_){
_start:
{
lean_object* v___x_884_; 
v___x_884_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg(v_msg_877_, v_declHint_878_, v___y_882_);
return v___x_884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___boxed(lean_object* v_msg_885_, lean_object* v_declHint_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_){
_start:
{
lean_object* v_res_892_; 
v_res_892_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6(v_msg_885_, v_declHint_886_, v___y_887_, v___y_888_, v___y_889_, v___y_890_);
lean_dec(v___y_890_);
lean_dec_ref(v___y_889_);
lean_dec(v___y_888_);
lean_dec_ref(v___y_887_);
return v_res_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(lean_object* v_00_u03b1_893_, lean_object* v_ref_894_, lean_object* v_msg_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_){
_start:
{
lean_object* v___x_901_; 
v___x_901_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___redArg(v_ref_894_, v_msg_895_, v___y_896_, v___y_897_, v___y_898_, v___y_899_);
return v___x_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6___boxed(lean_object* v_00_u03b1_902_, lean_object* v_ref_903_, lean_object* v_msg_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_){
_start:
{
lean_object* v_res_910_; 
v_res_910_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__6(v_00_u03b1_902_, v_ref_903_, v_msg_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_);
lean_dec(v___y_908_);
lean_dec_ref(v___y_907_);
lean_dec(v___y_906_);
lean_dec_ref(v___y_905_);
lean_dec(v_ref_903_);
return v_res_910_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; 
v___x_922_ = lean_box(0);
v___x_923_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_924_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_924_, 0, v___x_923_);
lean_ctor_set(v___x_924_, 1, v___x_922_);
return v___x_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg(){
_start:
{
lean_object* v___x_926_; lean_object* v___x_927_; 
v___x_926_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg___closed__0);
v___x_927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_927_, 0, v___x_926_);
return v___x_927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v___y_928_){
_start:
{
lean_object* v_res_929_; 
v_res_929_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg();
return v_res_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_930_, lean_object* v___y_931_, lean_object* v___y_932_){
_start:
{
lean_object* v___x_934_; 
v___x_934_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg();
return v___x_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_){
_start:
{
lean_object* v_res_939_; 
v_res_939_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1(v_00_u03b1_935_, v___y_936_, v___y_937_);
lean_dec(v___y_937_);
lean_dec_ref(v___y_936_);
return v_res_939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_msgData_940_, lean_object* v___y_941_, lean_object* v___y_942_){
_start:
{
lean_object* v___x_944_; lean_object* v_env_945_; lean_object* v_options_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; 
v___x_944_ = lean_st_ref_get(v___y_942_);
v_env_945_ = lean_ctor_get(v___x_944_, 0);
lean_inc_ref(v_env_945_);
lean_dec(v___x_944_);
v_options_946_ = lean_ctor_get(v___y_941_, 2);
v___x_947_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__2);
v___x_948_ = lean_unsigned_to_nat(32u);
v___x_949_ = lean_mk_empty_array_with_capacity(v___x_948_);
lean_dec_ref(v___x_949_);
v___x_950_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__5);
lean_inc_ref(v_options_946_);
v___x_951_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_951_, 0, v_env_945_);
lean_ctor_set(v___x_951_, 1, v___x_947_);
lean_ctor_set(v___x_951_, 2, v___x_950_);
lean_ctor_set(v___x_951_, 3, v_options_946_);
v___x_952_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_952_, 0, v___x_951_);
lean_ctor_set(v___x_952_, 1, v_msgData_940_);
v___x_953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_953_, 0, v___x_952_);
return v___x_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_msgData_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_){
_start:
{
lean_object* v_res_958_; 
v_res_958_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0_spec__0(v_msgData_954_, v___y_955_, v___y_956_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
return v_res_958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(lean_object* v_msg_959_, lean_object* v___y_960_, lean_object* v___y_961_){
_start:
{
lean_object* v_ref_963_; lean_object* v___x_964_; lean_object* v_a_965_; lean_object* v___x_967_; uint8_t v_isShared_968_; uint8_t v_isSharedCheck_973_; 
v_ref_963_ = lean_ctor_get(v___y_960_, 5);
v___x_964_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0_spec__0(v_msg_959_, v___y_960_, v___y_961_);
v_a_965_ = lean_ctor_get(v___x_964_, 0);
v_isSharedCheck_973_ = !lean_is_exclusive(v___x_964_);
if (v_isSharedCheck_973_ == 0)
{
v___x_967_ = v___x_964_;
v_isShared_968_ = v_isSharedCheck_973_;
goto v_resetjp_966_;
}
else
{
lean_inc(v_a_965_);
lean_dec(v___x_964_);
v___x_967_ = lean_box(0);
v_isShared_968_ = v_isSharedCheck_973_;
goto v_resetjp_966_;
}
v_resetjp_966_:
{
lean_object* v___x_969_; lean_object* v___x_971_; 
lean_inc(v_ref_963_);
v___x_969_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_969_, 0, v_ref_963_);
lean_ctor_set(v___x_969_, 1, v_a_965_);
if (v_isShared_968_ == 0)
{
lean_ctor_set_tag(v___x_967_, 1);
lean_ctor_set(v___x_967_, 0, v___x_969_);
v___x_971_ = v___x_967_;
goto v_reusejp_970_;
}
else
{
lean_object* v_reuseFailAlloc_972_; 
v_reuseFailAlloc_972_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_972_, 0, v___x_969_);
v___x_971_ = v_reuseFailAlloc_972_;
goto v_reusejp_970_;
}
v_reusejp_970_:
{
return v___x_971_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v_msg_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_){
_start:
{
lean_object* v_res_978_; 
v_res_978_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(v_msg_974_, v___y_975_, v___y_976_);
lean_dec(v___y_976_);
lean_dec_ref(v___y_975_);
return v_res_978_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_979_; 
v___x_979_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_979_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_980_; lean_object* v___x_981_; 
v___x_980_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__0, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__0);
v___x_981_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_981_, 0, v___x_980_);
return v___x_981_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_982_; lean_object* v___x_983_; 
v___x_982_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__1, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__1_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__1);
v___x_983_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_983_, 0, v___x_982_);
lean_ctor_set(v___x_983_, 1, v___x_982_);
return v___x_983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg(lean_object* v_env_984_, lean_object* v___y_985_){
_start:
{
lean_object* v___x_987_; lean_object* v_nextMacroScope_988_; lean_object* v_ngen_989_; lean_object* v_auxDeclNGen_990_; lean_object* v_traceState_991_; lean_object* v_messages_992_; lean_object* v_infoState_993_; lean_object* v_snapshotTasks_994_; lean_object* v___x_996_; uint8_t v_isShared_997_; uint8_t v_isSharedCheck_1005_; 
v___x_987_ = lean_st_ref_take(v___y_985_);
v_nextMacroScope_988_ = lean_ctor_get(v___x_987_, 1);
v_ngen_989_ = lean_ctor_get(v___x_987_, 2);
v_auxDeclNGen_990_ = lean_ctor_get(v___x_987_, 3);
v_traceState_991_ = lean_ctor_get(v___x_987_, 4);
v_messages_992_ = lean_ctor_get(v___x_987_, 6);
v_infoState_993_ = lean_ctor_get(v___x_987_, 7);
v_snapshotTasks_994_ = lean_ctor_get(v___x_987_, 8);
v_isSharedCheck_1005_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_1005_ == 0)
{
lean_object* v_unused_1006_; lean_object* v_unused_1007_; 
v_unused_1006_ = lean_ctor_get(v___x_987_, 5);
lean_dec(v_unused_1006_);
v_unused_1007_ = lean_ctor_get(v___x_987_, 0);
lean_dec(v_unused_1007_);
v___x_996_ = v___x_987_;
v_isShared_997_ = v_isSharedCheck_1005_;
goto v_resetjp_995_;
}
else
{
lean_inc(v_snapshotTasks_994_);
lean_inc(v_infoState_993_);
lean_inc(v_messages_992_);
lean_inc(v_traceState_991_);
lean_inc(v_auxDeclNGen_990_);
lean_inc(v_ngen_989_);
lean_inc(v_nextMacroScope_988_);
lean_dec(v___x_987_);
v___x_996_ = lean_box(0);
v_isShared_997_ = v_isSharedCheck_1005_;
goto v_resetjp_995_;
}
v_resetjp_995_:
{
lean_object* v___x_998_; lean_object* v___x_1000_; 
v___x_998_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___closed__2);
if (v_isShared_997_ == 0)
{
lean_ctor_set(v___x_996_, 5, v___x_998_);
lean_ctor_set(v___x_996_, 0, v_env_984_);
v___x_1000_ = v___x_996_;
goto v_reusejp_999_;
}
else
{
lean_object* v_reuseFailAlloc_1004_; 
v_reuseFailAlloc_1004_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1004_, 0, v_env_984_);
lean_ctor_set(v_reuseFailAlloc_1004_, 1, v_nextMacroScope_988_);
lean_ctor_set(v_reuseFailAlloc_1004_, 2, v_ngen_989_);
lean_ctor_set(v_reuseFailAlloc_1004_, 3, v_auxDeclNGen_990_);
lean_ctor_set(v_reuseFailAlloc_1004_, 4, v_traceState_991_);
lean_ctor_set(v_reuseFailAlloc_1004_, 5, v___x_998_);
lean_ctor_set(v_reuseFailAlloc_1004_, 6, v_messages_992_);
lean_ctor_set(v_reuseFailAlloc_1004_, 7, v_infoState_993_);
lean_ctor_set(v_reuseFailAlloc_1004_, 8, v_snapshotTasks_994_);
v___x_1000_ = v_reuseFailAlloc_1004_;
goto v_reusejp_999_;
}
v_reusejp_999_:
{
lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v___x_1001_ = lean_st_ref_set(v___y_985_, v___x_1000_);
v___x_1002_ = lean_box(0);
v___x_1003_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1003_, 0, v___x_1002_);
return v___x_1003_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg___boxed(lean_object* v_env_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
lean_object* v_res_1011_; 
v_res_1011_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg(v_env_1008_, v___y_1009_);
lean_dec(v___y_1009_);
return v_res_1011_;
}
}
static lean_object* _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1013_ = ((lean_object*)(lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__0));
v___x_1014_ = l_Lean_stringToMessageData(v___x_1013_);
return v___x_1014_;
}
}
static lean_object* _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; 
v___x_1016_ = ((lean_object*)(lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__2));
v___x_1017_ = l_Lean_stringToMessageData(v___x_1016_);
return v___x_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg(lean_object* v_ext_1018_, lean_object* v_k_1019_, lean_object* v_v_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_){
_start:
{
lean_object* v___x_1024_; lean_object* v_env_1025_; lean_object* v___x_1026_; 
v___x_1024_ = lean_st_ref_get(v___y_1022_);
v_env_1025_ = lean_ctor_get(v___x_1024_, 0);
lean_inc_ref(v_env_1025_);
lean_dec(v___x_1024_);
v___x_1026_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_1018_, v_env_1025_, v_k_1019_);
if (lean_obj_tag(v___x_1026_) == 1)
{
lean_object* v_name_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; 
lean_dec_ref_known(v___x_1026_, 1);
lean_dec(v_v_1020_);
v_name_1027_ = lean_ctor_get(v_ext_1018_, 1);
lean_inc(v_name_1027_);
lean_dec_ref(v_ext_1018_);
v___x_1028_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__1, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__1_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__1);
v___x_1029_ = l_Lean_MessageData_ofName(v_name_1027_);
v___x_1030_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1030_, 0, v___x_1028_);
lean_ctor_set(v___x_1030_, 1, v___x_1029_);
v___x_1031_ = lean_obj_once(&lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__3, &lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__3_once, _init_lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___closed__3);
v___x_1032_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1032_, 0, v___x_1030_);
lean_ctor_set(v___x_1032_, 1, v___x_1031_);
v___x_1033_ = l_Lean_MessageData_ofName(v_k_1019_);
v___x_1034_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1034_, 0, v___x_1032_);
lean_ctor_set(v___x_1034_, 1, v___x_1033_);
v___x_1035_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(v___x_1034_, v___y_1021_, v___y_1022_);
return v___x_1035_;
}
else
{
lean_object* v___x_1036_; lean_object* v_toEnvExtension_1037_; lean_object* v_env_1038_; lean_object* v_asyncMode_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; 
lean_dec(v___x_1026_);
v___x_1036_ = lean_st_ref_get(v___y_1022_);
v_toEnvExtension_1037_ = lean_ctor_get(v_ext_1018_, 0);
v_env_1038_ = lean_ctor_get(v___x_1036_, 0);
lean_inc_ref(v_env_1038_);
lean_dec(v___x_1036_);
v_asyncMode_1039_ = lean_ctor_get(v_toEnvExtension_1037_, 2);
lean_inc(v_asyncMode_1039_);
v___x_1040_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1040_, 0, v_k_1019_);
lean_ctor_set(v___x_1040_, 1, v_v_1020_);
v___x_1041_ = lean_box(0);
v___x_1042_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v_ext_1018_, v_env_1038_, v___x_1040_, v_asyncMode_1039_, v___x_1041_);
lean_dec(v_asyncMode_1039_);
v___x_1043_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg(v___x_1042_, v___y_1022_);
return v___x_1043_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object* v_ext_1044_, lean_object* v_k_1045_, lean_object* v_v_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_){
_start:
{
lean_object* v_res_1050_; 
v_res_1050_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg(v_ext_1044_, v_k_1045_, v_v_1046_, v___y_1047_, v___y_1048_);
lean_dec(v___y_1048_);
lean_dec_ref(v___y_1047_);
return v_res_1050_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1052_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1053_ = l_Lean_stringToMessageData(v___x_1052_);
return v___x_1053_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1055_; lean_object* v___x_1056_; 
v___x_1055_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1056_ = l_Lean_stringToMessageData(v___x_1055_);
return v___x_1056_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1058_; lean_object* v___x_1059_; 
v___x_1058_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__4_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1059_ = l_Lean_stringToMessageData(v___x_1058_);
return v___x_1059_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1060_; 
v___x_1060_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1060_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1061_; lean_object* v___x_1062_; 
v___x_1061_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__6_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1062_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1062_, 0, v___x_1061_);
return v___x_1062_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1063_; lean_object* v___x_1064_; 
v___x_1063_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1064_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1064_, 0, v___x_1063_);
lean_ctor_set(v___x_1064_, 1, v___x_1063_);
lean_ctor_set(v___x_1064_, 2, v___x_1063_);
lean_ctor_set(v___x_1064_, 3, v___x_1063_);
lean_ctor_set(v___x_1064_, 4, v___x_1063_);
lean_ctor_set(v___x_1064_, 5, v___x_1063_);
return v___x_1064_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1065_; lean_object* v___x_1066_; 
v___x_1065_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1066_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1066_, 0, v___x_1065_);
lean_ctor_set(v___x_1066_, 1, v___x_1065_);
lean_ctor_set(v___x_1066_, 2, v___x_1065_);
lean_ctor_set(v___x_1066_, 3, v___x_1065_);
lean_ctor_set(v___x_1066_, 4, v___x_1065_);
return v___x_1066_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__12_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1069_; lean_object* v___x_1070_; 
v___x_1069_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__11_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1070_ = l_Lean_stringToMessageData(v___x_1069_);
return v___x_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(lean_object* v_a_1071_, lean_object* v___x_1072_, lean_object* v___x_1073_, lean_object* v_src_1074_, lean_object* v_stx_1075_, uint8_t v___kind_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_){
_start:
{
lean_object* v___y_1081_; lean_object* v___y_1082_; lean_object* v___y_1083_; lean_object* v___y_1084_; uint8_t v___y_1085_; lean_object* v___y_1089_; uint8_t v___y_1090_; lean_object* v___y_1091_; lean_object* v___y_1092_; lean_object* v___y_1093_; lean_object* v___y_1094_; lean_object* v___y_1097_; lean_object* v___y_1098_; uint8_t v___y_1099_; lean_object* v___y_1100_; lean_object* v___y_1101_; lean_object* v___y_1102_; uint8_t v_a_1103_; lean_object* v___y_1111_; uint8_t v___y_1112_; lean_object* v___y_1113_; lean_object* v___y_1114_; lean_object* v___y_1115_; lean_object* v___y_1116_; lean_object* v___y_1117_; lean_object* v___y_1169_; uint8_t v___y_1170_; lean_object* v___y_1171_; lean_object* v___y_1172_; lean_object* v___y_1173_; lean_object* v___y_1174_; lean_object* v___y_1178_; uint8_t v___y_1179_; lean_object* v___y_1180_; lean_object* v___y_1181_; lean_object* v_projName_1182_; lean_object* v___y_1183_; lean_object* v___y_1184_; lean_object* v___y_1188_; uint8_t v___y_1189_; lean_object* v___y_1190_; lean_object* v___y_1191_; lean_object* v_findArgs_x3f_1192_; lean_object* v___y_1193_; lean_object* v___y_1194_; lean_object* v___y_1203_; uint8_t v___y_1204_; lean_object* v___y_1205_; lean_object* v___y_1206_; lean_object* v_projName_x3f_1207_; lean_object* v___y_1208_; lean_object* v___y_1209_; lean_object* v___y_1219_; uint8_t v___y_1220_; lean_object* v___y_1221_; lean_object* v_coercion_1222_; lean_object* v___y_1223_; lean_object* v___y_1224_; lean_object* v___y_1234_; lean_object* v___y_1235_; lean_object* v___x_1247_; lean_object* v_env_1248_; uint8_t v___x_1249_; 
v___x_1247_ = lean_st_ref_get(v___y_1078_);
v_env_1248_ = lean_ctor_get(v___x_1247_, 0);
lean_inc_ref(v_env_1248_);
lean_dec(v___x_1247_);
lean_inc(v_src_1074_);
v___x_1249_ = l_Lean_isStructure(v_env_1248_, v_src_1074_);
if (v___x_1249_ == 0)
{
lean_object* v___x_1250_; lean_object* v___x_1251_; 
lean_dec(v_stx_1075_);
lean_dec(v_src_1074_);
lean_dec_ref(v___x_1072_);
lean_dec_ref(v_a_1071_);
v___x_1250_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__12_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__12_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__12_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1251_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(v___x_1250_, v___y_1077_, v___y_1078_);
return v___x_1251_;
}
else
{
v___y_1234_ = v___y_1077_;
v___y_1235_ = v___y_1078_;
goto v___jp_1233_;
}
v___jp_1080_:
{
lean_object* v___x_1086_; lean_object* v___x_1087_; 
v___x_1086_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1086_, 0, v_src_1074_);
lean_ctor_set(v___x_1086_, 1, v___y_1082_);
lean_ctor_set_uint8(v___x_1086_, sizeof(void*)*2, v___y_1085_);
v___x_1087_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg(v_a_1071_, v___y_1084_, v___x_1086_, v___y_1083_, v___y_1081_);
return v___x_1087_;
}
v___jp_1088_:
{
if (lean_obj_tag(v___y_1091_) == 0)
{
v___y_1081_ = v___y_1094_;
v___y_1082_ = v___y_1089_;
v___y_1083_ = v___y_1093_;
v___y_1084_ = v___y_1092_;
v___y_1085_ = v___y_1090_;
goto v___jp_1080_;
}
else
{
uint8_t v___x_1095_; 
lean_dec_ref_known(v___y_1091_, 1);
v___x_1095_ = 0;
v___y_1081_ = v___y_1094_;
v___y_1082_ = v___y_1089_;
v___y_1083_ = v___y_1093_;
v___y_1084_ = v___y_1092_;
v___y_1085_ = v___x_1095_;
goto v___jp_1080_;
}
}
v___jp_1096_:
{
if (v_a_1103_ == 0)
{
lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; 
lean_dec(v___y_1102_);
lean_dec(v___y_1101_);
lean_dec(v_src_1074_);
lean_dec_ref(v_a_1071_);
v___x_1104_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1105_ = l_Lean_MessageData_ofName(v___y_1098_);
v___x_1106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1106_, 0, v___x_1104_);
lean_ctor_set(v___x_1106_, 1, v___x_1105_);
v___x_1107_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1108_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1108_, 0, v___x_1106_);
lean_ctor_set(v___x_1108_, 1, v___x_1107_);
v___x_1109_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(v___x_1108_, v___y_1097_, v___y_1100_);
return v___x_1109_;
}
else
{
v___y_1089_ = v___y_1098_;
v___y_1090_ = v___y_1099_;
v___y_1091_ = v___y_1101_;
v___y_1092_ = v___y_1102_;
v___y_1093_ = v___y_1097_;
v___y_1094_ = v___y_1100_;
goto v___jp_1088_;
}
}
v___jp_1110_:
{
lean_object* v___x_1118_; lean_object* v_env_1119_; uint8_t v___x_1120_; lean_object* v___x_1121_; 
v___x_1118_ = lean_st_ref_get(v___y_1114_);
v_env_1119_ = lean_ctor_get(v___x_1118_, 0);
lean_inc_ref(v_env_1119_);
lean_dec(v___x_1118_);
v___x_1120_ = 0;
lean_inc(v___y_1117_);
v___x_1121_ = l_Lean_Environment_find_x3f(v_env_1119_, v___y_1117_, v___x_1120_);
if (lean_obj_tag(v___x_1121_) == 0)
{
lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; 
lean_dec(v___y_1116_);
lean_dec(v___y_1115_);
lean_dec(v___y_1113_);
lean_dec(v_src_1074_);
lean_dec_ref(v___x_1072_);
lean_dec_ref(v_a_1071_);
v___x_1122_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__5_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1123_ = l_Lean_MessageData_ofName(v___y_1117_);
v___x_1124_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1124_, 0, v___x_1122_);
lean_ctor_set(v___x_1124_, 1, v___x_1123_);
v___x_1125_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(v___x_1124_, v___y_1111_, v___y_1114_);
return v___x_1125_;
}
else
{
lean_object* v_val_1126_; uint8_t v___x_1127_; uint8_t v___x_1128_; uint8_t v___x_1129_; lean_object* v___x_1130_; uint64_t v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; size_t v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; 
v_val_1126_ = lean_ctor_get(v___x_1121_, 0);
lean_inc(v_val_1126_);
lean_dec_ref_known(v___x_1121_, 1);
v___x_1127_ = 1;
v___x_1128_ = 0;
v___x_1129_ = 2;
v___x_1130_ = lean_alloc_ctor(0, 0, 20);
lean_ctor_set_uint8(v___x_1130_, 0, v___x_1120_);
lean_ctor_set_uint8(v___x_1130_, 1, v___x_1120_);
lean_ctor_set_uint8(v___x_1130_, 2, v___x_1120_);
lean_ctor_set_uint8(v___x_1130_, 3, v___x_1120_);
lean_ctor_set_uint8(v___x_1130_, 4, v___x_1120_);
lean_ctor_set_uint8(v___x_1130_, 5, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 6, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 7, v___x_1120_);
lean_ctor_set_uint8(v___x_1130_, 8, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 9, v___x_1127_);
lean_ctor_set_uint8(v___x_1130_, 10, v___x_1128_);
lean_ctor_set_uint8(v___x_1130_, 11, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 12, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 13, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 14, v___x_1129_);
lean_ctor_set_uint8(v___x_1130_, 15, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 16, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 17, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 18, v___y_1112_);
lean_ctor_set_uint8(v___x_1130_, 19, v___x_1120_);
v___x_1131_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1130_);
v___x_1132_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1132_, 0, v___x_1130_);
lean_ctor_set_uint64(v___x_1132_, sizeof(void*)*1, v___x_1131_);
v___x_1133_ = lean_box(1);
v___x_1134_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__7_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1135_ = lean_unsigned_to_nat(32u);
v___x_1136_ = lean_mk_empty_array_with_capacity(v___x_1135_);
v___x_1137_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Simps_findCoercionArgs_spec__0_spec__0_spec__1_spec__2_spec__4_spec__5_spec__6___redArg___closed__3);
v___x_1138_ = ((size_t)5ULL);
lean_inc_n(v___y_1113_, 6);
v___x_1139_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1139_, 0, v___x_1137_);
lean_ctor_set(v___x_1139_, 1, v___x_1136_);
lean_ctor_set(v___x_1139_, 2, v___y_1113_);
lean_ctor_set(v___x_1139_, 3, v___y_1113_);
lean_ctor_set_usize(v___x_1139_, 4, v___x_1138_);
lean_inc_ref(v___x_1139_);
v___x_1140_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1140_, 0, v___x_1134_);
lean_ctor_set(v___x_1140_, 1, v___x_1139_);
lean_ctor_set(v___x_1140_, 2, v___x_1133_);
v___x_1141_ = lean_mk_empty_array_with_capacity(v___y_1113_);
v___x_1142_ = lean_box(0);
v___x_1143_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1143_, 0, v___x_1132_);
lean_ctor_set(v___x_1143_, 1, v___x_1133_);
lean_ctor_set(v___x_1143_, 2, v___x_1140_);
lean_ctor_set(v___x_1143_, 3, v___x_1141_);
lean_ctor_set(v___x_1143_, 4, v___x_1142_);
lean_ctor_set(v___x_1143_, 5, v___y_1113_);
lean_ctor_set(v___x_1143_, 6, v___x_1142_);
lean_ctor_set_uint8(v___x_1143_, sizeof(void*)*7, v___x_1120_);
lean_ctor_set_uint8(v___x_1143_, sizeof(void*)*7 + 1, v___x_1120_);
lean_ctor_set_uint8(v___x_1143_, sizeof(void*)*7 + 2, v___x_1120_);
lean_ctor_set_uint8(v___x_1143_, sizeof(void*)*7 + 3, v___y_1112_);
v___x_1144_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1144_, 0, v___y_1113_);
lean_ctor_set(v___x_1144_, 1, v___y_1113_);
lean_ctor_set(v___x_1144_, 2, v___y_1113_);
lean_ctor_set(v___x_1144_, 3, v___y_1113_);
lean_ctor_set(v___x_1144_, 4, v___x_1134_);
lean_ctor_set(v___x_1144_, 5, v___x_1134_);
lean_ctor_set(v___x_1144_, 6, v___x_1134_);
lean_ctor_set(v___x_1144_, 7, v___x_1134_);
lean_ctor_set(v___x_1144_, 8, v___x_1134_);
lean_ctor_set(v___x_1144_, 9, v___x_1134_);
v___x_1145_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__8_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1146_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__9_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1147_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1147_, 0, v___x_1144_);
lean_ctor_set(v___x_1147_, 1, v___x_1145_);
lean_ctor_set(v___x_1147_, 2, v___x_1133_);
lean_ctor_set(v___x_1147_, 3, v___x_1139_);
lean_ctor_set(v___x_1147_, 4, v___x_1146_);
v___x_1148_ = lean_st_mk_ref(v___x_1147_);
v___x_1149_ = l_Lean_ConstantInfo_type(v_val_1126_);
lean_dec(v_val_1126_);
v___x_1150_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0___closed__10_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1151_ = l_Lean_Name_mkStr2(v___x_1072_, v___x_1150_);
v___x_1152_ = lean_box(0);
v___x_1153_ = l_Lean_mkConst(v___x_1151_, v___x_1152_);
v___x_1154_ = l_Lean_Meta_isExprDefEq(v___x_1149_, v___x_1153_, v___x_1143_, v___x_1148_, v___y_1111_, v___y_1114_);
lean_dec_ref_known(v___x_1143_, 7);
if (lean_obj_tag(v___x_1154_) == 0)
{
lean_object* v_a_1155_; lean_object* v___x_1156_; uint8_t v___x_1157_; 
v_a_1155_ = lean_ctor_get(v___x_1154_, 0);
lean_inc(v_a_1155_);
lean_dec_ref_known(v___x_1154_, 1);
v___x_1156_ = lean_st_ref_get(v___x_1148_);
lean_dec(v___x_1148_);
lean_dec(v___x_1156_);
v___x_1157_ = lean_unbox(v_a_1155_);
lean_dec(v_a_1155_);
v___y_1097_ = v___y_1111_;
v___y_1098_ = v___y_1117_;
v___y_1099_ = v___y_1112_;
v___y_1100_ = v___y_1114_;
v___y_1101_ = v___y_1115_;
v___y_1102_ = v___y_1116_;
v_a_1103_ = v___x_1157_;
goto v___jp_1096_;
}
else
{
lean_dec(v___x_1148_);
if (lean_obj_tag(v___x_1154_) == 0)
{
lean_object* v_a_1158_; uint8_t v___x_1159_; 
v_a_1158_ = lean_ctor_get(v___x_1154_, 0);
lean_inc(v_a_1158_);
lean_dec_ref_known(v___x_1154_, 1);
v___x_1159_ = lean_unbox(v_a_1158_);
lean_dec(v_a_1158_);
v___y_1097_ = v___y_1111_;
v___y_1098_ = v___y_1117_;
v___y_1099_ = v___y_1112_;
v___y_1100_ = v___y_1114_;
v___y_1101_ = v___y_1115_;
v___y_1102_ = v___y_1116_;
v_a_1103_ = v___x_1159_;
goto v___jp_1096_;
}
else
{
lean_object* v_a_1160_; lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1167_; 
lean_dec(v___y_1117_);
lean_dec(v___y_1116_);
lean_dec(v___y_1115_);
lean_dec(v_src_1074_);
lean_dec_ref(v_a_1071_);
v_a_1160_ = lean_ctor_get(v___x_1154_, 0);
v_isSharedCheck_1167_ = !lean_is_exclusive(v___x_1154_);
if (v_isSharedCheck_1167_ == 0)
{
v___x_1162_ = v___x_1154_;
v_isShared_1163_ = v_isSharedCheck_1167_;
goto v_resetjp_1161_;
}
else
{
lean_inc(v_a_1160_);
lean_dec(v___x_1154_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1167_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v___x_1165_; 
if (v_isShared_1163_ == 0)
{
v___x_1165_ = v___x_1162_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v_a_1160_);
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
v___jp_1168_:
{
lean_object* v___x_1175_; lean_object* v___x_1176_; 
v___x_1175_ = ((lean_object*)(lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__1));
lean_inc_ref(v___x_1072_);
v___x_1176_ = l_Lean_Name_mkStr2(v___x_1072_, v___x_1175_);
v___y_1111_ = v___y_1169_;
v___y_1112_ = v___y_1170_;
v___y_1113_ = v___y_1172_;
v___y_1114_ = v___y_1171_;
v___y_1115_ = v___y_1173_;
v___y_1116_ = v___y_1174_;
v___y_1117_ = v___x_1176_;
goto v___jp_1110_;
}
v___jp_1177_:
{
if (lean_obj_tag(v___y_1178_) == 0)
{
v___y_1169_ = v___y_1183_;
v___y_1170_ = v___y_1179_;
v___y_1171_ = v___y_1184_;
v___y_1172_ = v___y_1180_;
v___y_1173_ = v___y_1181_;
v___y_1174_ = v_projName_1182_;
goto v___jp_1168_;
}
else
{
if (v___y_1179_ == 0)
{
lean_dec_ref_known(v___y_1178_, 1);
v___y_1169_ = v___y_1183_;
v___y_1170_ = v___y_1179_;
v___y_1171_ = v___y_1184_;
v___y_1172_ = v___y_1180_;
v___y_1173_ = v___y_1181_;
v___y_1174_ = v_projName_1182_;
goto v___jp_1168_;
}
else
{
lean_object* v_val_1185_; lean_object* v___x_1186_; 
v_val_1185_ = lean_ctor_get(v___y_1178_, 0);
lean_inc(v_val_1185_);
lean_dec_ref_known(v___y_1178_, 1);
v___x_1186_ = l_Lean_TSyntax_getId(v_val_1185_);
lean_dec(v_val_1185_);
v___y_1111_ = v___y_1183_;
v___y_1112_ = v___y_1179_;
v___y_1113_ = v___y_1180_;
v___y_1114_ = v___y_1184_;
v___y_1115_ = v___y_1181_;
v___y_1116_ = v_projName_1182_;
v___y_1117_ = v___x_1186_;
goto v___jp_1110_;
}
}
}
v___jp_1187_:
{
if (lean_obj_tag(v___y_1188_) == 0)
{
lean_object* v___x_1195_; lean_object* v_env_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; 
v___x_1195_ = lean_st_ref_get(v___y_1194_);
v_env_1196_ = lean_ctor_get(v___x_1195_, 0);
lean_inc_ref(v_env_1196_);
lean_dec(v___x_1195_);
v___x_1197_ = lean_box(0);
lean_inc(v_src_1074_);
v___x_1198_ = l_Lean_getStructureFields(v_env_1196_, v_src_1074_);
v___x_1199_ = lean_array_get(v___x_1197_, v___x_1198_, v___y_1190_);
lean_dec_ref(v___x_1198_);
v___y_1178_ = v_findArgs_x3f_1192_;
v___y_1179_ = v___y_1189_;
v___y_1180_ = v___y_1190_;
v___y_1181_ = v___y_1191_;
v_projName_1182_ = v___x_1199_;
v___y_1183_ = v___y_1193_;
v___y_1184_ = v___y_1194_;
goto v___jp_1177_;
}
else
{
lean_object* v_val_1200_; lean_object* v___x_1201_; 
v_val_1200_ = lean_ctor_get(v___y_1188_, 0);
lean_inc(v_val_1200_);
lean_dec_ref_known(v___y_1188_, 1);
v___x_1201_ = l_Lean_TSyntax_getId(v_val_1200_);
lean_dec(v_val_1200_);
v___y_1178_ = v_findArgs_x3f_1192_;
v___y_1179_ = v___y_1189_;
v___y_1180_ = v___y_1190_;
v___y_1181_ = v___y_1191_;
v_projName_1182_ = v___x_1201_;
v___y_1183_ = v___y_1193_;
v___y_1184_ = v___y_1194_;
goto v___jp_1177_;
}
}
v___jp_1202_:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; uint8_t v___x_1212_; 
v___x_1210_ = lean_unsigned_to_nat(3u);
v___x_1211_ = l_Lean_Syntax_getArg(v_stx_1075_, v___x_1210_);
lean_dec(v_stx_1075_);
v___x_1212_ = l_Lean_Syntax_isNone(v___x_1211_);
if (v___x_1212_ == 0)
{
uint8_t v___x_1213_; 
lean_inc(v___x_1211_);
v___x_1213_ = l_Lean_Syntax_matchesNull(v___x_1211_, v___y_1203_);
if (v___x_1213_ == 0)
{
lean_object* v___x_1214_; 
lean_dec(v___x_1211_);
lean_dec(v_projName_x3f_1207_);
lean_dec(v___y_1206_);
lean_dec(v___y_1205_);
lean_dec(v_src_1074_);
lean_dec_ref(v___x_1072_);
lean_dec_ref(v_a_1071_);
v___x_1214_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg();
return v___x_1214_;
}
else
{
lean_object* v___x_1215_; lean_object* v___x_1216_; 
v___x_1215_ = l_Lean_Syntax_getArg(v___x_1211_, v___y_1205_);
lean_dec(v___x_1211_);
v___x_1216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1216_, 0, v___x_1215_);
v___y_1188_ = v_projName_x3f_1207_;
v___y_1189_ = v___y_1204_;
v___y_1190_ = v___y_1205_;
v___y_1191_ = v___y_1206_;
v_findArgs_x3f_1192_ = v___x_1216_;
v___y_1193_ = v___y_1208_;
v___y_1194_ = v___y_1209_;
goto v___jp_1187_;
}
}
else
{
lean_object* v___x_1217_; 
lean_dec(v___x_1211_);
v___x_1217_ = lean_box(0);
v___y_1188_ = v_projName_x3f_1207_;
v___y_1189_ = v___y_1204_;
v___y_1190_ = v___y_1205_;
v___y_1191_ = v___y_1206_;
v_findArgs_x3f_1192_ = v___x_1217_;
v___y_1193_ = v___y_1208_;
v___y_1194_ = v___y_1209_;
goto v___jp_1187_;
}
}
v___jp_1218_:
{
lean_object* v___x_1225_; lean_object* v___x_1226_; uint8_t v___x_1227_; 
v___x_1225_ = lean_unsigned_to_nat(2u);
v___x_1226_ = l_Lean_Syntax_getArg(v_stx_1075_, v___x_1225_);
v___x_1227_ = l_Lean_Syntax_isNone(v___x_1226_);
if (v___x_1227_ == 0)
{
uint8_t v___x_1228_; 
lean_inc(v___x_1226_);
v___x_1228_ = l_Lean_Syntax_matchesNull(v___x_1226_, v___y_1219_);
if (v___x_1228_ == 0)
{
lean_object* v___x_1229_; 
lean_dec(v___x_1226_);
lean_dec(v_coercion_1222_);
lean_dec(v___y_1221_);
lean_dec(v_stx_1075_);
lean_dec(v_src_1074_);
lean_dec_ref(v___x_1072_);
lean_dec_ref(v_a_1071_);
v___x_1229_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg();
return v___x_1229_;
}
else
{
lean_object* v___x_1230_; lean_object* v___x_1231_; 
v___x_1230_ = l_Lean_Syntax_getArg(v___x_1226_, v___y_1221_);
lean_dec(v___x_1226_);
v___x_1231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1231_, 0, v___x_1230_);
v___y_1203_ = v___y_1219_;
v___y_1204_ = v___y_1220_;
v___y_1205_ = v___y_1221_;
v___y_1206_ = v_coercion_1222_;
v_projName_x3f_1207_ = v___x_1231_;
v___y_1208_ = v___y_1223_;
v___y_1209_ = v___y_1224_;
goto v___jp_1202_;
}
}
else
{
lean_object* v___x_1232_; 
lean_dec(v___x_1226_);
v___x_1232_ = lean_box(0);
v___y_1203_ = v___y_1219_;
v___y_1204_ = v___y_1220_;
v___y_1205_ = v___y_1221_;
v___y_1206_ = v_coercion_1222_;
v_projName_x3f_1207_ = v___x_1232_;
v___y_1208_ = v___y_1223_;
v___y_1209_ = v___y_1224_;
goto v___jp_1202_;
}
}
v___jp_1233_:
{
uint8_t v___x_1236_; 
lean_inc(v_stx_1075_);
v___x_1236_ = l_Lean_Syntax_isOfKind(v_stx_1075_, v___x_1073_);
if (v___x_1236_ == 0)
{
lean_object* v___x_1237_; 
lean_dec(v_stx_1075_);
lean_dec(v_src_1074_);
lean_dec_ref(v___x_1072_);
lean_dec_ref(v_a_1071_);
v___x_1237_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg();
return v___x_1237_;
}
else
{
lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; uint8_t v___x_1241_; 
v___x_1238_ = lean_unsigned_to_nat(0u);
v___x_1239_ = lean_unsigned_to_nat(1u);
v___x_1240_ = l_Lean_Syntax_getArg(v_stx_1075_, v___x_1239_);
v___x_1241_ = l_Lean_Syntax_isNone(v___x_1240_);
if (v___x_1241_ == 0)
{
uint8_t v___x_1242_; 
lean_inc(v___x_1240_);
v___x_1242_ = l_Lean_Syntax_matchesNull(v___x_1240_, v___x_1239_);
if (v___x_1242_ == 0)
{
lean_object* v___x_1243_; 
lean_dec(v___x_1240_);
lean_dec(v_stx_1075_);
lean_dec(v_src_1074_);
lean_dec_ref(v___x_1072_);
lean_dec_ref(v_a_1071_);
v___x_1243_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__1___redArg();
return v___x_1243_;
}
else
{
lean_object* v___x_1244_; lean_object* v___x_1245_; 
v___x_1244_ = l_Lean_Syntax_getArg(v___x_1240_, v___x_1238_);
lean_dec(v___x_1240_);
v___x_1245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1245_, 0, v___x_1244_);
v___y_1219_ = v___x_1239_;
v___y_1220_ = v___x_1236_;
v___y_1221_ = v___x_1238_;
v_coercion_1222_ = v___x_1245_;
v___y_1223_ = v___y_1234_;
v___y_1224_ = v___y_1235_;
goto v___jp_1218_;
}
}
else
{
lean_object* v___x_1246_; 
lean_dec(v___x_1240_);
v___x_1246_ = lean_box(0);
v___y_1219_ = v___x_1239_;
v___y_1220_ = v___x_1236_;
v___y_1221_ = v___x_1238_;
v_coercion_1222_ = v___x_1246_;
v___y_1223_ = v___y_1234_;
v___y_1224_ = v___y_1235_;
goto v___jp_1218_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2____boxed(lean_object* v_a_1252_, lean_object* v___x_1253_, lean_object* v___x_1254_, lean_object* v_src_1255_, lean_object* v_stx_1256_, lean_object* v___kind_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_){
_start:
{
uint8_t v___kind_boxed_1261_; lean_object* v_res_1262_; 
v___kind_boxed_1261_ = lean_unbox(v___kind_1257_);
v_res_1262_ = lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(v_a_1252_, v___x_1253_, v___x_1254_, v_src_1255_, v_stx_1256_, v___kind_boxed_1261_, v___y_1258_, v___y_1259_);
lean_dec(v___y_1259_);
lean_dec_ref(v___y_1258_);
lean_dec(v___x_1254_);
return v_res_1262_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1264_; lean_object* v___x_1265_; 
v___x_1264_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1265_ = l_Lean_stringToMessageData(v___x_1264_);
return v___x_1265_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1267_; lean_object* v___x_1268_; 
v___x_1267_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1268_ = l_Lean_stringToMessageData(v___x_1267_);
return v___x_1268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(lean_object* v___x_1269_, lean_object* v_decl_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_){
_start:
{
lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; 
v___x_1274_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1275_ = l_Lean_MessageData_ofName(v___x_1269_);
v___x_1276_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1276_, 0, v___x_1274_);
lean_ctor_set(v___x_1276_, 1, v___x_1275_);
v___x_1277_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_);
v___x_1278_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1278_, 0, v___x_1276_);
lean_ctor_set(v___x_1278_, 1, v___x_1277_);
v___x_1279_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(v___x_1278_, v___y_1271_, v___y_1272_);
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2____boxed(lean_object* v___x_1280_, lean_object* v_decl_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_){
_start:
{
lean_object* v_res_1285_; 
v_res_1285_ = lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(v___x_1280_, v_decl_1281_, v___y_1282_, v___y_1283_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec(v_decl_1281_);
return v_res_1285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; 
v___x_1299_ = ((lean_object*)(lp_mathlib_Simps_instInhabitedAutomaticProjectionData_default___closed__0));
v___x_1300_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__1_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1301_ = lp_batteries_Lean_registerNameMapExtension___redArg(v___x_1300_);
if (lean_obj_tag(v___x_1301_) == 0)
{
lean_object* v_a_1302_; lean_object* v___x_1303_; lean_object* v___f_1304_; lean_object* v___f_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; 
v_a_1302_ = lean_ctor_get(v___x_1301_, 0);
lean_inc_n(v_a_1302_, 2);
lean_dec_ref_known(v___x_1301_, 1);
v___x_1303_ = ((lean_object*)(lp_mathlib_notation__class___closed__1));
v___f_1304_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___lam__0_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2____boxed), 9, 3);
lean_closure_set(v___f_1304_, 0, v_a_1302_);
lean_closure_set(v___f_1304_, 1, v___x_1299_);
lean_closure_set(v___f_1304_, 2, v___x_1303_);
v___f_1305_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__2_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1306_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn___closed__4_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_));
v___x_1307_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1307_, 0, v___x_1306_);
lean_ctor_set(v___x_1307_, 1, v___f_1304_);
lean_ctor_set(v___x_1307_, 2, v___f_1305_);
v___x_1308_ = l_Lean_registerBuiltinAttribute(v___x_1307_);
if (lean_obj_tag(v___x_1308_) == 0)
{
lean_object* v___x_1310_; uint8_t v_isShared_1311_; uint8_t v_isSharedCheck_1315_; 
v_isSharedCheck_1315_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1315_ == 0)
{
lean_object* v_unused_1316_; 
v_unused_1316_ = lean_ctor_get(v___x_1308_, 0);
lean_dec(v_unused_1316_);
v___x_1310_ = v___x_1308_;
v_isShared_1311_ = v_isSharedCheck_1315_;
goto v_resetjp_1309_;
}
else
{
lean_dec(v___x_1308_);
v___x_1310_ = lean_box(0);
v_isShared_1311_ = v_isSharedCheck_1315_;
goto v_resetjp_1309_;
}
v_resetjp_1309_:
{
lean_object* v___x_1313_; 
if (v_isShared_1311_ == 0)
{
lean_ctor_set(v___x_1310_, 0, v_a_1302_);
v___x_1313_ = v___x_1310_;
goto v_reusejp_1312_;
}
else
{
lean_object* v_reuseFailAlloc_1314_; 
v_reuseFailAlloc_1314_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1314_, 0, v_a_1302_);
v___x_1313_ = v_reuseFailAlloc_1314_;
goto v_reusejp_1312_;
}
v_reusejp_1312_:
{
return v___x_1313_;
}
}
}
else
{
lean_object* v_a_1317_; lean_object* v___x_1319_; uint8_t v_isShared_1320_; uint8_t v_isSharedCheck_1324_; 
lean_dec(v_a_1302_);
v_a_1317_ = lean_ctor_get(v___x_1308_, 0);
v_isSharedCheck_1324_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1324_ == 0)
{
v___x_1319_ = v___x_1308_;
v_isShared_1320_ = v_isSharedCheck_1324_;
goto v_resetjp_1318_;
}
else
{
lean_inc(v_a_1317_);
lean_dec(v___x_1308_);
v___x_1319_ = lean_box(0);
v_isShared_1320_ = v_isSharedCheck_1324_;
goto v_resetjp_1318_;
}
v_resetjp_1318_:
{
lean_object* v___x_1322_; 
if (v_isShared_1320_ == 0)
{
v___x_1322_ = v___x_1319_;
goto v_reusejp_1321_;
}
else
{
lean_object* v_reuseFailAlloc_1323_; 
v_reuseFailAlloc_1323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1323_, 0, v_a_1317_);
v___x_1322_ = v_reuseFailAlloc_1323_;
goto v_reusejp_1321_;
}
v_reusejp_1321_:
{
return v___x_1322_;
}
}
}
}
else
{
return v___x_1301_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2____boxed(lean_object* v_a_1325_){
_start:
{
lean_object* v_res_1326_; 
v_res_1326_ = lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_();
return v_res_1326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_1327_, lean_object* v_msg_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_){
_start:
{
lean_object* v___x_1332_; 
v___x_1332_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___redArg(v_msg_1328_, v___y_1329_, v___y_1330_);
return v___x_1332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_1333_, lean_object* v_msg_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_){
_start:
{
lean_object* v_res_1338_; 
v_res_1338_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__0(v_00_u03b1_1333_, v_msg_1334_, v___y_1335_, v___y_1336_);
lean_dec(v___y_1336_);
lean_dec_ref(v___y_1335_);
return v_res_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3(lean_object* v_env_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
lean_object* v___x_1343_; 
v___x_1343_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___redArg(v_env_1339_, v___y_1341_);
return v___x_1343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3___boxed(lean_object* v_env_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_){
_start:
{
lean_object* v_res_1348_; 
v_res_1348_ = lp_mathlib_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2_spec__3(v_env_1344_, v___y_1345_, v___y_1346_);
lean_dec(v___y_1346_);
lean_dec_ref(v___y_1345_);
return v_res_1348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2(lean_object* v_00_u03b1_1349_, lean_object* v_ext_1350_, lean_object* v_k_1351_, lean_object* v_v_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_){
_start:
{
lean_object* v___x_1356_; 
v___x_1356_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___redArg(v_ext_1350_, v_k_1351_, v_v_1352_, v___y_1353_, v___y_1354_);
return v___x_1356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2___boxed(lean_object* v_00_u03b1_1357_, lean_object* v_ext_1358_, lean_object* v_k_1359_, lean_object* v_v_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_){
_start:
{
lean_object* v_res_1364_; 
v_res_1364_ = lp_mathlib_Lean_NameMapExtension_add___at___00__private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2__spec__2(v_00_u03b1_1357_, v_ext_1358_, v_k_1359_, v_v_1360_, v___y_1361_, v___y_1362_);
lean_dec(v___y_1362_);
lean_dec_ref(v___y_1361_);
return v_res_1364_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps_NotationClass(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Exception(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_NameMapAttribute(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Simps_NotationClass(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_NameMapAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Simps_NotationClass_0__Simps_initFn_00___x40_Mathlib_Tactic_Simps_NotationClass_2094931376____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Simps_notationClassAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Simps_notationClassAttr);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Exception(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_NameMapAttribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Simps_NotationClass(uint8_t builtin) {
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
res = initialize_Lean_Elab_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_NameMapAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps_NotationClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Simps_NotationClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Simps_NotationClass(builtin);
}
#ifdef __cplusplus
}
#endif
