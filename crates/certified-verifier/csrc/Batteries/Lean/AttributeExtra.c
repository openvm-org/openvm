// Lean compiler output
// Module: Batteries.Lean.AttributeExtra
// Imports: public import Init public meta import Init public import Batteries.Lean.TagAttribute public import Std.Data.HashMap.Basic
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
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_registerParametricAttribute___redArg(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_instInhabitedEnvExtension_default(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ParametricAttribute_getParam_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ParametricAttribute_setParam___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_Name_quickLt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_registerTagAttribute(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_NameHashSet_insert(lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedParametricAttribute_default(lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
uint8_t l_Lean_NameHashSet_contains(lean_object*, lean_object*);
lean_object* l_Lean_PersistentEnvExtension_getModuleEntries___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_instInhabitedPersistentEnvExtensionState___redArg(lean_object*);
lean_object* l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_TagAttribute_getDecls_core(lean_object*);
static const lean_string_object lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "(`Inhabited.default` for `IO.Error`)"};
static const lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___closed__0 = (const lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 18}, .m_objs = {((lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___closed__0_value)}};
static const lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___closed__1 = (const lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__1___boxed(lean_object*, lean_object*);
static const lean_array_object lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__0 = (const lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__0_value),((lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__0_value),((lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__0_value)}};
static const lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__1 = (const lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__3___boxed(lean_object*);
static const lean_closure_object lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__0 = (const lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__0_value;
static const lean_closure_object lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__1 = (const lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__1_value;
static const lean_closure_object lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__2 = (const lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__2_value;
static const lean_closure_object lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__3 = (const lean_object*)&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__3_value;
static lean_once_cell_t lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__4;
static lean_once_cell_t lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__5;
static lean_once_cell_t lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__6;
static lean_once_cell_t lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7;
static lean_once_cell_t lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__8;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__0 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__0_value;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__1 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__1_value;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__2 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__2_value;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__3 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__3_value;
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4_value_aux_0),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4_value_aux_1),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4_value_aux_2),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4_value;
static const lean_array_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__5 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__5_value;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__6 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__6_value;
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7_value_aux_0),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7_value_aux_1),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7_value_aux_2),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7_value;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__8 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__8_value;
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__9 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__9_value;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__10 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__10_value;
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11_value_aux_0),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11_value_aux_1),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11_value_aux_2),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11_value;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__12;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__13;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__14 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__14_value;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "declName"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__15 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__15_value;
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16_value_aux_0),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16_value_aux_1),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16_value_aux_2),((lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(113, 211, 58, 33, 138, 196, 138, 106)}};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16_value;
static const lean_string_object lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "decl_name%"};
static const lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__17 = (const lean_object*)&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__17_value;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__18;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__19;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__20;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__21;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__22;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__23;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__24;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__25;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__26;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__27;
static lean_once_cell_t lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__28;
LEAN_EXPORT lean_object* lp_batteries_Lean_registerTagAttributeExtra___auto__1;
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_registerTagAttributeExtra_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerTagAttributeExtra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerTagAttributeExtra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_TagAttributeExtra_hasTag(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttributeExtra_hasTag___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Lean_TagAttributeExtra_getDecls_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_TagAttributeExtra_getDecls_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_TagAttributeExtra_getDecls_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_TagAttributeExtra_getDecls___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_TagAttributeExtra_getDecls___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttributeExtra_getDecls(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttributeExtra_getDecls___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__0;
static lean_once_cell_t lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedParametricAttributeExtra_default(lean_object*);
static lean_once_cell_t lp_batteries_Lean_instInhabitedParametricAttributeExtra___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedParametricAttributeExtra___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedParametricAttributeExtra(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_registerParametricAttributeExtra_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerParametricAttributeExtra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerParametricAttributeExtra___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerParametricAttributeExtra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerParametricAttributeExtra___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_registerParametricAttributeExtra_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_setParam___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_setParam(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0(lean_object* v_x_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = ((lean_object*)(lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___closed__1));
v___x_8_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0___boxed(lean_object* v_x_9_, lean_object* v___y_10_, lean_object* v___y_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__0(v_x_9_, v___y_10_);
lean_dec_ref(v___y_10_);
lean_dec_ref(v_x_9_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__1(lean_object* v_s_13_, lean_object* v_x_14_){
_start:
{
lean_inc(v_s_13_);
return v_s_13_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__1___boxed(lean_object* v_s_15_, lean_object* v_x_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__1(v_s_15_, v_x_16_);
lean_dec(v_x_16_);
lean_dec(v_s_15_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2(lean_object* v_x_22_, lean_object* v_x_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = ((lean_object*)(lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___closed__1));
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2___boxed(lean_object* v_x_25_, lean_object* v_x_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__2(v_x_25_, v_x_26_);
lean_dec(v_x_26_);
lean_dec_ref(v_x_25_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__3(lean_object* v_x_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lean_box(0);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__3___boxed(lean_object* v_x_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_batteries_Lean_instInhabitedTagAttributeExtra_default___lam__3(v_x_30_);
lean_dec(v_x_30_);
return v_res_31_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__4(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = l_Lean_instInhabitedEnvExtension_default(lean_box(0));
return v___x_36_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__5(void){
_start:
{
lean_object* v___f_37_; lean_object* v___f_38_; lean_object* v___f_39_; lean_object* v___f_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___f_37_ = ((lean_object*)(lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__3));
v___f_38_ = ((lean_object*)(lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__2));
v___f_39_ = ((lean_object*)(lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__1));
v___f_40_ = ((lean_object*)(lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__0));
v___x_41_ = lean_box(0);
v___x_42_ = lean_obj_once(&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__4, &lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__4_once, _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__4);
v___x_43_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_43_, 0, v___x_42_);
lean_ctor_set(v___x_43_, 1, v___x_41_);
lean_ctor_set(v___x_43_, 2, v___f_40_);
lean_ctor_set(v___x_43_, 3, v___f_39_);
lean_ctor_set(v___x_43_, 4, v___f_38_);
lean_ctor_set(v___x_43_, 5, v___f_37_);
return v___x_43_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__6(void){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_44_ = lean_box(0);
v___x_45_ = lean_unsigned_to_nat(16u);
v___x_46_ = lean_mk_array(v___x_45_, v___x_44_);
return v___x_46_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_47_ = lean_obj_once(&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__6, &lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__6_once, _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__6);
v___x_48_ = lean_unsigned_to_nat(0u);
v___x_49_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_49_, 0, v___x_48_);
lean_ctor_set(v___x_49_, 1, v___x_47_);
return v___x_49_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__8(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_50_ = lean_obj_once(&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7, &lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7_once, _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7);
v___x_51_ = lean_obj_once(&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__5, &lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__5_once, _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__5);
v___x_52_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
lean_ctor_set(v___x_52_, 1, v___x_50_);
return v___x_52_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default(void){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lean_obj_once(&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__8, &lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__8_once, _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__8);
return v___x_53_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedTagAttributeExtra(void){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_batteries_Lean_instInhabitedTagAttributeExtra_default;
return v___x_54_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__12(void){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_81_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__10));
v___x_82_ = l_Lean_mkAtom(v___x_81_);
return v___x_82_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__13(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_83_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__12, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__12_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__12);
v___x_84_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__5));
v___x_85_ = lean_array_push(v___x_84_, v___x_83_);
return v___x_85_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__18(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__17));
v___x_95_ = l_Lean_mkAtom(v___x_94_);
return v___x_95_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__19(void){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_96_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__18, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__18_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__18);
v___x_97_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__5));
v___x_98_ = lean_array_push(v___x_97_, v___x_96_);
return v___x_98_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__20(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_99_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__19, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__19_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__19);
v___x_100_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__16));
v___x_101_ = lean_box(2);
v___x_102_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v___x_100_);
lean_ctor_set(v___x_102_, 2, v___x_99_);
return v___x_102_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__21(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__20, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__20_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__20);
v___x_104_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__13, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__13_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__13);
v___x_105_ = lean_array_push(v___x_104_, v___x_103_);
return v___x_105_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__22(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__21, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__21_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__21);
v___x_107_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__11));
v___x_108_ = lean_box(2);
v___x_109_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v___x_107_);
lean_ctor_set(v___x_109_, 2, v___x_106_);
return v___x_109_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__23(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_110_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__22, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__22_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__22);
v___x_111_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__5));
v___x_112_ = lean_array_push(v___x_111_, v___x_110_);
return v___x_112_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__24(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_113_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__23, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__23_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__23);
v___x_114_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__9));
v___x_115_ = lean_box(2);
v___x_116_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v___x_114_);
lean_ctor_set(v___x_116_, 2, v___x_113_);
return v___x_116_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__25(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_117_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__24, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__24_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__24);
v___x_118_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__5));
v___x_119_ = lean_array_push(v___x_118_, v___x_117_);
return v___x_119_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__26(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_120_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__25, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__25_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__25);
v___x_121_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__7));
v___x_122_ = lean_box(2);
v___x_123_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v___x_121_);
lean_ctor_set(v___x_123_, 2, v___x_120_);
return v___x_123_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__27(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_124_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__26, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__26_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__26);
v___x_125_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__5));
v___x_126_ = lean_array_push(v___x_125_, v___x_124_);
return v___x_126_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__28(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_127_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__27, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__27_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__27);
v___x_128_ = ((lean_object*)(lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__4));
v___x_129_ = lean_box(2);
v___x_130_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
lean_ctor_set(v___x_130_, 1, v___x_128_);
lean_ctor_set(v___x_130_, 2, v___x_127_);
return v___x_130_;
}
}
static lean_object* _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1(void){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lean_obj_once(&lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__28, &lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__28_once, _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1___closed__28);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_registerTagAttributeExtra_spec__0(lean_object* v_x_132_, lean_object* v_x_133_){
_start:
{
if (lean_obj_tag(v_x_133_) == 0)
{
return v_x_132_;
}
else
{
lean_object* v_head_134_; lean_object* v_tail_135_; lean_object* v___x_136_; 
v_head_134_ = lean_ctor_get(v_x_133_, 0);
lean_inc(v_head_134_);
v_tail_135_ = lean_ctor_get(v_x_133_, 1);
lean_inc(v_tail_135_);
lean_dec_ref_known(v_x_133_, 2);
v___x_136_ = l_Lean_NameHashSet_insert(v_x_132_, v_head_134_);
v_x_132_ = v___x_136_;
v_x_133_ = v_tail_135_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerTagAttributeExtra(lean_object* v_name_138_, lean_object* v_descr_139_, lean_object* v_extra_140_, lean_object* v_validate_141_, lean_object* v_ref_142_){
_start:
{
uint8_t v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
v___x_144_ = 0;
v___x_145_ = lean_box(2);
v___x_146_ = l_Lean_registerTagAttribute(v_name_138_, v_descr_139_, v_validate_141_, v_ref_142_, v___x_144_, v___x_145_);
if (lean_obj_tag(v___x_146_) == 0)
{
lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_165_; 
v_a_147_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_165_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_165_ == 0)
{
v___x_149_ = v___x_146_;
v_isShared_150_ = v_isSharedCheck_165_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_146_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_165_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v_ext_151_; lean_object* v___x_153_; uint8_t v_isShared_154_; uint8_t v_isSharedCheck_163_; 
v_ext_151_ = lean_ctor_get(v_a_147_, 1);
v_isSharedCheck_163_ = !lean_is_exclusive(v_a_147_);
if (v_isSharedCheck_163_ == 0)
{
lean_object* v_unused_164_; 
v_unused_164_ = lean_ctor_get(v_a_147_, 0);
lean_dec(v_unused_164_);
v___x_153_ = v_a_147_;
v_isShared_154_ = v_isSharedCheck_163_;
goto v_resetjp_152_;
}
else
{
lean_inc(v_ext_151_);
lean_dec(v_a_147_);
v___x_153_ = lean_box(0);
v_isShared_154_ = v_isSharedCheck_163_;
goto v_resetjp_152_;
}
v_resetjp_152_:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_158_; 
v___x_155_ = lean_obj_once(&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7, &lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7_once, _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7);
v___x_156_ = lp_batteries_List_foldl___at___00Lean_registerTagAttributeExtra_spec__0(v___x_155_, v_extra_140_);
if (v_isShared_154_ == 0)
{
lean_ctor_set(v___x_153_, 1, v___x_156_);
lean_ctor_set(v___x_153_, 0, v_ext_151_);
v___x_158_ = v___x_153_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_ext_151_);
lean_ctor_set(v_reuseFailAlloc_162_, 1, v___x_156_);
v___x_158_ = v_reuseFailAlloc_162_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
lean_object* v___x_160_; 
if (v_isShared_150_ == 0)
{
lean_ctor_set(v___x_149_, 0, v___x_158_);
v___x_160_ = v___x_149_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v___x_158_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
}
else
{
lean_object* v_a_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_173_; 
lean_dec(v_extra_140_);
v_a_166_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_173_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_173_ == 0)
{
v___x_168_ = v___x_146_;
v_isShared_169_ = v_isSharedCheck_173_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_a_166_);
lean_dec(v___x_146_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_173_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v___x_171_; 
if (v_isShared_169_ == 0)
{
v___x_171_ = v___x_168_;
goto v_reusejp_170_;
}
else
{
lean_object* v_reuseFailAlloc_172_; 
v_reuseFailAlloc_172_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_172_, 0, v_a_166_);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_registerTagAttributeExtra___boxed(lean_object* v_name_174_, lean_object* v_descr_175_, lean_object* v_extra_176_, lean_object* v_validate_177_, lean_object* v_ref_178_, lean_object* v_a_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_batteries_Lean_registerTagAttributeExtra(v_name_174_, v_descr_175_, v_extra_176_, v_validate_177_, v_ref_178_);
return v_res_180_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___redArg(lean_object* v_as_181_, lean_object* v_k_182_, lean_object* v_x_183_, lean_object* v_x_184_){
_start:
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v_m_187_; lean_object* v_a_188_; uint8_t v___x_189_; 
v___x_185_ = lean_nat_add(v_x_183_, v_x_184_);
v___x_186_ = lean_unsigned_to_nat(1u);
v_m_187_ = lean_nat_shiftr(v___x_185_, v___x_186_);
lean_dec(v___x_185_);
v_a_188_ = lean_array_fget_borrowed(v_as_181_, v_m_187_);
v___x_189_ = l_Lean_Name_quickLt(v_a_188_, v_k_182_);
if (v___x_189_ == 0)
{
uint8_t v___x_190_; 
lean_dec(v_x_184_);
v___x_190_ = l_Lean_Name_quickLt(v_k_182_, v_a_188_);
if (v___x_190_ == 0)
{
uint8_t v___x_191_; 
lean_dec(v_m_187_);
lean_dec(v_x_183_);
v___x_191_ = 1;
return v___x_191_;
}
else
{
lean_object* v___x_192_; uint8_t v___x_193_; 
v___x_192_ = lean_unsigned_to_nat(0u);
v___x_193_ = lean_nat_dec_eq(v_m_187_, v___x_192_);
if (v___x_193_ == 0)
{
lean_object* v___x_194_; uint8_t v___x_195_; 
v___x_194_ = lean_nat_sub(v_m_187_, v___x_186_);
lean_dec(v_m_187_);
v___x_195_ = lean_nat_dec_lt(v___x_194_, v_x_183_);
if (v___x_195_ == 0)
{
v_x_184_ = v___x_194_;
goto _start;
}
else
{
lean_dec(v___x_194_);
lean_dec(v_x_183_);
return v___x_189_;
}
}
else
{
lean_dec(v_m_187_);
lean_dec(v_x_183_);
return v___x_189_;
}
}
}
else
{
lean_object* v___x_197_; uint8_t v___x_198_; 
lean_dec(v_x_183_);
v___x_197_ = lean_nat_add(v_m_187_, v___x_186_);
lean_dec(v_m_187_);
v___x_198_ = lean_nat_dec_le(v___x_197_, v_x_184_);
if (v___x_198_ == 0)
{
lean_dec(v___x_197_);
lean_dec(v_x_184_);
return v___x_198_;
}
else
{
v_x_183_ = v___x_197_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___redArg___boxed(lean_object* v_as_200_, lean_object* v_k_201_, lean_object* v_x_202_, lean_object* v_x_203_){
_start:
{
uint8_t v_res_204_; lean_object* v_r_205_; 
v_res_204_ = lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___redArg(v_as_200_, v_k_201_, v_x_202_, v_x_203_);
lean_dec(v_k_201_);
lean_dec_ref(v_as_200_);
v_r_205_ = lean_box(v_res_204_);
return v_r_205_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_TagAttributeExtra_hasTag(lean_object* v_attr_206_, lean_object* v_env_207_, lean_object* v_decl_208_){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = lean_box(1);
v___x_210_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_207_, v_decl_208_);
if (lean_obj_tag(v___x_210_) == 0)
{
lean_object* v_ext_211_; lean_object* v_toEnvExtension_212_; lean_object* v_base_213_; lean_object* v_asyncMode_214_; lean_object* v___x_215_; lean_object* v___x_216_; uint8_t v___x_217_; 
v_ext_211_ = lean_ctor_get(v_attr_206_, 0);
v_toEnvExtension_212_ = lean_ctor_get(v_ext_211_, 0);
v_base_213_ = lean_ctor_get(v_attr_206_, 1);
v_asyncMode_214_ = lean_ctor_get(v_toEnvExtension_212_, 2);
v___x_215_ = lean_box(0);
v___x_216_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_209_, v_ext_211_, v_env_207_, v_asyncMode_214_, v___x_215_);
v___x_217_ = l_Lean_NameSet_contains(v___x_216_, v_decl_208_);
lean_dec(v___x_216_);
if (v___x_217_ == 0)
{
uint8_t v___x_218_; 
v___x_218_ = l_Lean_NameHashSet_contains(v_base_213_, v_decl_208_);
return v___x_218_;
}
else
{
return v___x_217_;
}
}
else
{
lean_object* v_val_219_; lean_object* v_ext_220_; uint8_t v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v_val_219_ = lean_ctor_get(v___x_210_, 0);
lean_inc(v_val_219_);
lean_dec_ref_known(v___x_210_, 1);
v_ext_220_ = lean_ctor_get(v_attr_206_, 0);
v___x_221_ = 0;
v___x_222_ = l_Lean_PersistentEnvExtension_getModuleEntries___redArg(v___x_209_, v_ext_220_, v_env_207_, v_val_219_, v___x_221_);
lean_dec(v_val_219_);
lean_dec_ref(v_env_207_);
v___x_223_ = lean_unsigned_to_nat(0u);
v___x_224_ = lean_array_get_size(v___x_222_);
v___x_225_ = lean_nat_dec_lt(v___x_223_, v___x_224_);
if (v___x_225_ == 0)
{
lean_dec_ref(v___x_222_);
return v___x_225_;
}
else
{
lean_object* v___x_226_; lean_object* v___x_227_; uint8_t v___x_228_; 
v___x_226_ = lean_unsigned_to_nat(1u);
v___x_227_ = lean_nat_sub(v___x_224_, v___x_226_);
v___x_228_ = lean_nat_dec_le(v___x_223_, v___x_227_);
if (v___x_228_ == 0)
{
lean_dec(v___x_227_);
lean_dec_ref(v___x_222_);
return v___x_228_;
}
else
{
uint8_t v___x_229_; 
v___x_229_ = lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___redArg(v___x_222_, v_decl_208_, v___x_223_, v___x_227_);
lean_dec_ref(v___x_222_);
return v___x_229_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttributeExtra_hasTag___boxed(lean_object* v_attr_230_, lean_object* v_env_231_, lean_object* v_decl_232_){
_start:
{
uint8_t v_res_233_; lean_object* v_r_234_; 
v_res_233_ = lp_batteries_Lean_TagAttributeExtra_hasTag(v_attr_230_, v_env_231_, v_decl_232_);
lean_dec(v_decl_232_);
lean_dec_ref(v_attr_230_);
v_r_234_ = lean_box(v_res_233_);
return v_r_234_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0(lean_object* v_as_235_, lean_object* v_k_236_, lean_object* v_x_237_, lean_object* v_x_238_, lean_object* v_x_239_){
_start:
{
uint8_t v___x_240_; 
v___x_240_ = lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___redArg(v_as_235_, v_k_236_, v_x_237_, v_x_238_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0___boxed(lean_object* v_as_241_, lean_object* v_k_242_, lean_object* v_x_243_, lean_object* v_x_244_, lean_object* v_x_245_){
_start:
{
uint8_t v_res_246_; lean_object* v_r_247_; 
v_res_246_ = lp_batteries_Array_binSearchAux___at___00Lean_TagAttributeExtra_hasTag_spec__0(v_as_241_, v_k_242_, v_x_243_, v_x_244_, v_x_245_);
lean_dec(v_k_242_);
lean_dec_ref(v_as_241_);
v_r_247_ = lean_box(v_res_246_);
return v_r_247_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Lean_TagAttributeExtra_getDecls_spec__0(lean_object* v_x_248_, lean_object* v_x_249_){
_start:
{
if (lean_obj_tag(v_x_249_) == 0)
{
return v_x_248_;
}
else
{
lean_object* v_key_250_; lean_object* v_tail_251_; lean_object* v___x_252_; 
v_key_250_ = lean_ctor_get(v_x_249_, 0);
lean_inc(v_key_250_);
v_tail_251_ = lean_ctor_get(v_x_249_, 2);
lean_inc(v_tail_251_);
lean_dec_ref_known(v_x_249_, 3);
v___x_252_ = lean_array_push(v_x_248_, v_key_250_);
v_x_248_ = v___x_252_;
v_x_249_ = v_tail_251_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_TagAttributeExtra_getDecls_spec__1(lean_object* v_as_254_, size_t v_i_255_, size_t v_stop_256_, lean_object* v_b_257_){
_start:
{
uint8_t v___x_258_; 
v___x_258_ = lean_usize_dec_eq(v_i_255_, v_stop_256_);
if (v___x_258_ == 0)
{
lean_object* v___x_259_; lean_object* v___x_260_; size_t v___x_261_; size_t v___x_262_; 
v___x_259_ = lean_array_uget_borrowed(v_as_254_, v_i_255_);
lean_inc(v___x_259_);
v___x_260_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Lean_TagAttributeExtra_getDecls_spec__0(v_b_257_, v___x_259_);
v___x_261_ = ((size_t)1ULL);
v___x_262_ = lean_usize_add(v_i_255_, v___x_261_);
v_i_255_ = v___x_262_;
v_b_257_ = v___x_260_;
goto _start;
}
else
{
return v_b_257_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_TagAttributeExtra_getDecls_spec__1___boxed(lean_object* v_as_264_, lean_object* v_i_265_, lean_object* v_stop_266_, lean_object* v_b_267_){
_start:
{
size_t v_i_boxed_268_; size_t v_stop_boxed_269_; lean_object* v_res_270_; 
v_i_boxed_268_ = lean_unbox_usize(v_i_265_);
lean_dec(v_i_265_);
v_stop_boxed_269_ = lean_unbox_usize(v_stop_266_);
lean_dec(v_stop_266_);
v_res_270_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_TagAttributeExtra_getDecls_spec__1(v_as_264_, v_i_boxed_268_, v_stop_boxed_269_, v_b_267_);
lean_dec_ref(v_as_264_);
return v_res_270_;
}
}
static lean_object* _init_lp_batteries_Lean_TagAttributeExtra_getDecls___closed__0(void){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_271_ = lean_box(1);
v___x_272_ = l_Lean_instInhabitedPersistentEnvExtensionState___redArg(v___x_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttributeExtra_getDecls(lean_object* v_attr_273_, lean_object* v_env_274_){
_start:
{
lean_object* v_ext_275_; lean_object* v_toEnvExtension_276_; lean_object* v_base_277_; lean_object* v_asyncMode_278_; lean_object* v_buckets_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v_decls_283_; lean_object* v___x_284_; lean_object* v___x_285_; uint8_t v___x_286_; 
v_ext_275_ = lean_ctor_get(v_attr_273_, 0);
v_toEnvExtension_276_ = lean_ctor_get(v_ext_275_, 0);
v_base_277_ = lean_ctor_get(v_attr_273_, 1);
v_asyncMode_278_ = lean_ctor_get(v_toEnvExtension_276_, 2);
v_buckets_279_ = lean_ctor_get(v_base_277_, 1);
v___x_280_ = lean_obj_once(&lp_batteries_Lean_TagAttributeExtra_getDecls___closed__0, &lp_batteries_Lean_TagAttributeExtra_getDecls___closed__0_once, _init_lp_batteries_Lean_TagAttributeExtra_getDecls___closed__0);
v___x_281_ = lean_box(0);
v___x_282_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_280_, v_toEnvExtension_276_, v_env_274_, v_asyncMode_278_, v___x_281_);
v_decls_283_ = lp_batteries_Lean_TagAttribute_getDecls_core(v___x_282_);
v___x_284_ = lean_unsigned_to_nat(0u);
v___x_285_ = lean_array_get_size(v_buckets_279_);
v___x_286_ = lean_nat_dec_lt(v___x_284_, v___x_285_);
if (v___x_286_ == 0)
{
return v_decls_283_;
}
else
{
uint8_t v___x_287_; 
v___x_287_ = lean_nat_dec_le(v___x_285_, v___x_285_);
if (v___x_287_ == 0)
{
if (v___x_286_ == 0)
{
return v_decls_283_;
}
else
{
size_t v___x_288_; size_t v___x_289_; lean_object* v___x_290_; 
v___x_288_ = ((size_t)0ULL);
v___x_289_ = lean_usize_of_nat(v___x_285_);
v___x_290_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_TagAttributeExtra_getDecls_spec__1(v_buckets_279_, v___x_288_, v___x_289_, v_decls_283_);
return v___x_290_;
}
}
else
{
size_t v___x_291_; size_t v___x_292_; lean_object* v___x_293_; 
v___x_291_ = ((size_t)0ULL);
v___x_292_ = lean_usize_of_nat(v___x_285_);
v___x_293_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_TagAttributeExtra_getDecls_spec__1(v_buckets_279_, v___x_291_, v___x_292_, v_decls_283_);
return v___x_293_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttributeExtra_getDecls___boxed(lean_object* v_attr_294_, lean_object* v_env_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_batteries_Lean_TagAttributeExtra_getDecls(v_attr_294_, v_env_295_);
lean_dec_ref(v_attr_294_);
return v_res_296_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__0(void){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = l_Lean_instInhabitedParametricAttribute_default(lean_box(0));
return v___x_297_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__1(void){
_start:
{
lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_298_ = lean_obj_once(&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7, &lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7_once, _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7);
v___x_299_ = lean_obj_once(&lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__0, &lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__0_once, _init_lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__0);
v___x_300_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_300_, 0, v___x_299_);
lean_ctor_set(v___x_300_, 1, v___x_298_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedParametricAttributeExtra_default(lean_object* v_00_u03b1_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lean_obj_once(&lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__1, &lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__1_once, _init_lp_batteries_Lean_instInhabitedParametricAttributeExtra_default___closed__1);
return v___x_302_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedParametricAttributeExtra___closed__0(void){
_start:
{
lean_object* v___x_303_; 
v___x_303_ = lp_batteries_Lean_instInhabitedParametricAttributeExtra_default(lean_box(0));
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedParametricAttributeExtra(lean_object* v_a_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lean_obj_once(&lp_batteries_Lean_instInhabitedParametricAttributeExtra___closed__0, &lp_batteries_Lean_instInhabitedParametricAttributeExtra___closed__0_once, _init_lp_batteries_Lean_instInhabitedParametricAttributeExtra___closed__0);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__2___redArg(lean_object* v_a_306_, lean_object* v_b_307_, lean_object* v_x_308_){
_start:
{
if (lean_obj_tag(v_x_308_) == 0)
{
lean_dec(v_b_307_);
lean_dec(v_a_306_);
return v_x_308_;
}
else
{
lean_object* v_key_309_; lean_object* v_value_310_; lean_object* v_tail_311_; lean_object* v___x_313_; uint8_t v_isShared_314_; uint8_t v_isSharedCheck_323_; 
v_key_309_ = lean_ctor_get(v_x_308_, 0);
v_value_310_ = lean_ctor_get(v_x_308_, 1);
v_tail_311_ = lean_ctor_get(v_x_308_, 2);
v_isSharedCheck_323_ = !lean_is_exclusive(v_x_308_);
if (v_isSharedCheck_323_ == 0)
{
v___x_313_ = v_x_308_;
v_isShared_314_ = v_isSharedCheck_323_;
goto v_resetjp_312_;
}
else
{
lean_inc(v_tail_311_);
lean_inc(v_value_310_);
lean_inc(v_key_309_);
lean_dec(v_x_308_);
v___x_313_ = lean_box(0);
v_isShared_314_ = v_isSharedCheck_323_;
goto v_resetjp_312_;
}
v_resetjp_312_:
{
uint8_t v___x_315_; 
v___x_315_ = lean_name_eq(v_key_309_, v_a_306_);
if (v___x_315_ == 0)
{
lean_object* v___x_316_; lean_object* v___x_318_; 
v___x_316_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__2___redArg(v_a_306_, v_b_307_, v_tail_311_);
if (v_isShared_314_ == 0)
{
lean_ctor_set(v___x_313_, 2, v___x_316_);
v___x_318_ = v___x_313_;
goto v_reusejp_317_;
}
else
{
lean_object* v_reuseFailAlloc_319_; 
v_reuseFailAlloc_319_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_319_, 0, v_key_309_);
lean_ctor_set(v_reuseFailAlloc_319_, 1, v_value_310_);
lean_ctor_set(v_reuseFailAlloc_319_, 2, v___x_316_);
v___x_318_ = v_reuseFailAlloc_319_;
goto v_reusejp_317_;
}
v_reusejp_317_:
{
return v___x_318_;
}
}
else
{
lean_object* v___x_321_; 
lean_dec(v_value_310_);
lean_dec(v_key_309_);
if (v_isShared_314_ == 0)
{
lean_ctor_set(v___x_313_, 1, v_b_307_);
lean_ctor_set(v___x_313_, 0, v_a_306_);
v___x_321_ = v___x_313_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_a_306_);
lean_ctor_set(v_reuseFailAlloc_322_, 1, v_b_307_);
lean_ctor_set(v_reuseFailAlloc_322_, 2, v_tail_311_);
v___x_321_ = v_reuseFailAlloc_322_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
return v___x_321_;
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___redArg(lean_object* v_a_324_, lean_object* v_x_325_){
_start:
{
if (lean_obj_tag(v_x_325_) == 0)
{
uint8_t v___x_326_; 
v___x_326_ = 0;
return v___x_326_;
}
else
{
lean_object* v_key_327_; lean_object* v_tail_328_; uint8_t v___x_329_; 
v_key_327_ = lean_ctor_get(v_x_325_, 0);
v_tail_328_ = lean_ctor_get(v_x_325_, 2);
v___x_329_ = lean_name_eq(v_key_327_, v_a_324_);
if (v___x_329_ == 0)
{
v_x_325_ = v_tail_328_;
goto _start;
}
else
{
return v___x_329_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___redArg___boxed(lean_object* v_a_331_, lean_object* v_x_332_){
_start:
{
uint8_t v_res_333_; lean_object* v_r_334_; 
v_res_333_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___redArg(v_a_331_, v_x_332_);
lean_dec(v_x_332_);
lean_dec(v_a_331_);
v_r_334_ = lean_box(v_res_333_);
return v_r_334_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2_spec__4___redArg(lean_object* v_x_335_, lean_object* v_x_336_){
_start:
{
if (lean_obj_tag(v_x_336_) == 0)
{
return v_x_335_;
}
else
{
lean_object* v_key_337_; lean_object* v_value_338_; lean_object* v_tail_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_365_; 
v_key_337_ = lean_ctor_get(v_x_336_, 0);
v_value_338_ = lean_ctor_get(v_x_336_, 1);
v_tail_339_ = lean_ctor_get(v_x_336_, 2);
v_isSharedCheck_365_ = !lean_is_exclusive(v_x_336_);
if (v_isSharedCheck_365_ == 0)
{
v___x_341_ = v_x_336_;
v_isShared_342_ = v_isSharedCheck_365_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_tail_339_);
lean_inc(v_value_338_);
lean_inc(v_key_337_);
lean_dec(v_x_336_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_365_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v___x_343_; uint64_t v___y_345_; 
v___x_343_ = lean_array_get_size(v_x_335_);
if (lean_obj_tag(v_key_337_) == 0)
{
uint64_t v___x_363_; 
v___x_363_ = 1723ULL;
v___y_345_ = v___x_363_;
goto v___jp_344_;
}
else
{
uint64_t v_hash_364_; 
v_hash_364_ = lean_ctor_get_uint64(v_key_337_, sizeof(void*)*2);
v___y_345_ = v_hash_364_;
goto v___jp_344_;
}
v___jp_344_:
{
uint64_t v___x_346_; uint64_t v___x_347_; uint64_t v_fold_348_; uint64_t v___x_349_; uint64_t v___x_350_; uint64_t v___x_351_; size_t v___x_352_; size_t v___x_353_; size_t v___x_354_; size_t v___x_355_; size_t v___x_356_; lean_object* v___x_357_; lean_object* v___x_359_; 
v___x_346_ = 32ULL;
v___x_347_ = lean_uint64_shift_right(v___y_345_, v___x_346_);
v_fold_348_ = lean_uint64_xor(v___y_345_, v___x_347_);
v___x_349_ = 16ULL;
v___x_350_ = lean_uint64_shift_right(v_fold_348_, v___x_349_);
v___x_351_ = lean_uint64_xor(v_fold_348_, v___x_350_);
v___x_352_ = lean_uint64_to_usize(v___x_351_);
v___x_353_ = lean_usize_of_nat(v___x_343_);
v___x_354_ = ((size_t)1ULL);
v___x_355_ = lean_usize_sub(v___x_353_, v___x_354_);
v___x_356_ = lean_usize_land(v___x_352_, v___x_355_);
v___x_357_ = lean_array_uget_borrowed(v_x_335_, v___x_356_);
lean_inc(v___x_357_);
if (v_isShared_342_ == 0)
{
lean_ctor_set(v___x_341_, 2, v___x_357_);
v___x_359_ = v___x_341_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v_key_337_);
lean_ctor_set(v_reuseFailAlloc_362_, 1, v_value_338_);
lean_ctor_set(v_reuseFailAlloc_362_, 2, v___x_357_);
v___x_359_ = v_reuseFailAlloc_362_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
lean_object* v___x_360_; 
v___x_360_ = lean_array_uset(v_x_335_, v___x_356_, v___x_359_);
v_x_335_ = v___x_360_;
v_x_336_ = v_tail_339_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2___redArg(lean_object* v_i_366_, lean_object* v_source_367_, lean_object* v_target_368_){
_start:
{
lean_object* v___x_369_; uint8_t v___x_370_; 
v___x_369_ = lean_array_get_size(v_source_367_);
v___x_370_ = lean_nat_dec_lt(v_i_366_, v___x_369_);
if (v___x_370_ == 0)
{
lean_dec_ref(v_source_367_);
lean_dec(v_i_366_);
return v_target_368_;
}
else
{
lean_object* v_es_371_; lean_object* v___x_372_; lean_object* v_source_373_; lean_object* v_target_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v_es_371_ = lean_array_fget(v_source_367_, v_i_366_);
v___x_372_ = lean_box(0);
v_source_373_ = lean_array_fset(v_source_367_, v_i_366_, v___x_372_);
v_target_374_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2_spec__4___redArg(v_target_368_, v_es_371_);
v___x_375_ = lean_unsigned_to_nat(1u);
v___x_376_ = lean_nat_add(v_i_366_, v___x_375_);
lean_dec(v_i_366_);
v_i_366_ = v___x_376_;
v_source_367_ = v_source_373_;
v_target_368_ = v_target_374_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1___redArg(lean_object* v_data_378_){
_start:
{
lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v_nbuckets_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; 
v___x_379_ = lean_array_get_size(v_data_378_);
v___x_380_ = lean_unsigned_to_nat(2u);
v_nbuckets_381_ = lean_nat_mul(v___x_379_, v___x_380_);
v___x_382_ = lean_unsigned_to_nat(0u);
v___x_383_ = lean_box(0);
v___x_384_ = lean_mk_array(v_nbuckets_381_, v___x_383_);
v___x_385_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2___redArg(v___x_382_, v_data_378_, v___x_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0___redArg(lean_object* v_m_386_, lean_object* v_a_387_, lean_object* v_b_388_){
_start:
{
lean_object* v_size_389_; lean_object* v_buckets_390_; lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_436_; 
v_size_389_ = lean_ctor_get(v_m_386_, 0);
v_buckets_390_ = lean_ctor_get(v_m_386_, 1);
v_isSharedCheck_436_ = !lean_is_exclusive(v_m_386_);
if (v_isSharedCheck_436_ == 0)
{
v___x_392_ = v_m_386_;
v_isShared_393_ = v_isSharedCheck_436_;
goto v_resetjp_391_;
}
else
{
lean_inc(v_buckets_390_);
lean_inc(v_size_389_);
lean_dec(v_m_386_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_436_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
lean_object* v___x_394_; uint64_t v___y_396_; 
v___x_394_ = lean_array_get_size(v_buckets_390_);
if (lean_obj_tag(v_a_387_) == 0)
{
uint64_t v___x_434_; 
v___x_434_ = 1723ULL;
v___y_396_ = v___x_434_;
goto v___jp_395_;
}
else
{
uint64_t v_hash_435_; 
v_hash_435_ = lean_ctor_get_uint64(v_a_387_, sizeof(void*)*2);
v___y_396_ = v_hash_435_;
goto v___jp_395_;
}
v___jp_395_:
{
uint64_t v___x_397_; uint64_t v___x_398_; uint64_t v_fold_399_; uint64_t v___x_400_; uint64_t v___x_401_; uint64_t v___x_402_; size_t v___x_403_; size_t v___x_404_; size_t v___x_405_; size_t v___x_406_; size_t v___x_407_; lean_object* v_bkt_408_; uint8_t v___x_409_; 
v___x_397_ = 32ULL;
v___x_398_ = lean_uint64_shift_right(v___y_396_, v___x_397_);
v_fold_399_ = lean_uint64_xor(v___y_396_, v___x_398_);
v___x_400_ = 16ULL;
v___x_401_ = lean_uint64_shift_right(v_fold_399_, v___x_400_);
v___x_402_ = lean_uint64_xor(v_fold_399_, v___x_401_);
v___x_403_ = lean_uint64_to_usize(v___x_402_);
v___x_404_ = lean_usize_of_nat(v___x_394_);
v___x_405_ = ((size_t)1ULL);
v___x_406_ = lean_usize_sub(v___x_404_, v___x_405_);
v___x_407_ = lean_usize_land(v___x_403_, v___x_406_);
v_bkt_408_ = lean_array_uget_borrowed(v_buckets_390_, v___x_407_);
v___x_409_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___redArg(v_a_387_, v_bkt_408_);
if (v___x_409_ == 0)
{
lean_object* v___x_410_; lean_object* v_size_x27_411_; lean_object* v___x_412_; lean_object* v_buckets_x27_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; uint8_t v___x_419_; 
v___x_410_ = lean_unsigned_to_nat(1u);
v_size_x27_411_ = lean_nat_add(v_size_389_, v___x_410_);
lean_dec(v_size_389_);
lean_inc(v_bkt_408_);
v___x_412_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_412_, 0, v_a_387_);
lean_ctor_set(v___x_412_, 1, v_b_388_);
lean_ctor_set(v___x_412_, 2, v_bkt_408_);
v_buckets_x27_413_ = lean_array_uset(v_buckets_390_, v___x_407_, v___x_412_);
v___x_414_ = lean_unsigned_to_nat(4u);
v___x_415_ = lean_nat_mul(v_size_x27_411_, v___x_414_);
v___x_416_ = lean_unsigned_to_nat(3u);
v___x_417_ = lean_nat_div(v___x_415_, v___x_416_);
lean_dec(v___x_415_);
v___x_418_ = lean_array_get_size(v_buckets_x27_413_);
v___x_419_ = lean_nat_dec_le(v___x_417_, v___x_418_);
lean_dec(v___x_417_);
if (v___x_419_ == 0)
{
lean_object* v_val_420_; lean_object* v___x_422_; 
v_val_420_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1___redArg(v_buckets_x27_413_);
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 1, v_val_420_);
lean_ctor_set(v___x_392_, 0, v_size_x27_411_);
v___x_422_ = v___x_392_;
goto v_reusejp_421_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v_size_x27_411_);
lean_ctor_set(v_reuseFailAlloc_423_, 1, v_val_420_);
v___x_422_ = v_reuseFailAlloc_423_;
goto v_reusejp_421_;
}
v_reusejp_421_:
{
return v___x_422_;
}
}
else
{
lean_object* v___x_425_; 
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 1, v_buckets_x27_413_);
lean_ctor_set(v___x_392_, 0, v_size_x27_411_);
v___x_425_ = v___x_392_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v_size_x27_411_);
lean_ctor_set(v_reuseFailAlloc_426_, 1, v_buckets_x27_413_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
else
{
lean_object* v___x_427_; lean_object* v_buckets_x27_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_432_; 
lean_inc(v_bkt_408_);
v___x_427_ = lean_box(0);
v_buckets_x27_428_ = lean_array_uset(v_buckets_390_, v___x_407_, v___x_427_);
v___x_429_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__2___redArg(v_a_387_, v_b_388_, v_bkt_408_);
v___x_430_ = lean_array_uset(v_buckets_x27_428_, v___x_407_, v___x_429_);
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 1, v___x_430_);
v___x_432_ = v___x_392_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v_size_389_);
lean_ctor_set(v_reuseFailAlloc_433_, 1, v___x_430_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
return v___x_432_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_registerParametricAttributeExtra_spec__1___redArg(lean_object* v_x_437_, lean_object* v_x_438_){
_start:
{
if (lean_obj_tag(v_x_438_) == 0)
{
return v_x_437_;
}
else
{
lean_object* v_head_439_; lean_object* v_tail_440_; lean_object* v_fst_441_; lean_object* v_snd_442_; lean_object* v___x_443_; 
v_head_439_ = lean_ctor_get(v_x_438_, 0);
lean_inc(v_head_439_);
v_tail_440_ = lean_ctor_get(v_x_438_, 1);
lean_inc(v_tail_440_);
lean_dec_ref_known(v_x_438_, 2);
v_fst_441_ = lean_ctor_get(v_head_439_, 0);
lean_inc(v_fst_441_);
v_snd_442_ = lean_ctor_get(v_head_439_, 1);
lean_inc(v_snd_442_);
lean_dec(v_head_439_);
v___x_443_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0___redArg(v_x_437_, v_fst_441_, v_snd_442_);
v_x_437_ = v___x_443_;
v_x_438_ = v_tail_440_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerParametricAttributeExtra___redArg(lean_object* v_impl_445_, lean_object* v_extra_446_){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = l_Lean_registerParametricAttribute___redArg(v_impl_445_);
if (lean_obj_tag(v___x_448_) == 0)
{
lean_object* v_a_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_459_; 
v_a_449_ = lean_ctor_get(v___x_448_, 0);
v_isSharedCheck_459_ = !lean_is_exclusive(v___x_448_);
if (v_isSharedCheck_459_ == 0)
{
v___x_451_ = v___x_448_;
v_isShared_452_ = v_isSharedCheck_459_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_a_449_);
lean_dec(v___x_448_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_459_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_457_; 
v___x_453_ = lean_obj_once(&lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7, &lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7_once, _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default___closed__7);
v___x_454_ = lp_batteries_List_foldl___at___00Lean_registerParametricAttributeExtra_spec__1___redArg(v___x_453_, v_extra_446_);
v___x_455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_455_, 0, v_a_449_);
lean_ctor_set(v___x_455_, 1, v___x_454_);
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 0, v___x_455_);
v___x_457_ = v___x_451_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v___x_455_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
else
{
lean_object* v_a_460_; lean_object* v___x_462_; uint8_t v_isShared_463_; uint8_t v_isSharedCheck_467_; 
lean_dec(v_extra_446_);
v_a_460_ = lean_ctor_get(v___x_448_, 0);
v_isSharedCheck_467_ = !lean_is_exclusive(v___x_448_);
if (v_isSharedCheck_467_ == 0)
{
v___x_462_ = v___x_448_;
v_isShared_463_ = v_isSharedCheck_467_;
goto v_resetjp_461_;
}
else
{
lean_inc(v_a_460_);
lean_dec(v___x_448_);
v___x_462_ = lean_box(0);
v_isShared_463_ = v_isSharedCheck_467_;
goto v_resetjp_461_;
}
v_resetjp_461_:
{
lean_object* v___x_465_; 
if (v_isShared_463_ == 0)
{
v___x_465_ = v___x_462_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_466_; 
v_reuseFailAlloc_466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_466_, 0, v_a_460_);
v___x_465_ = v_reuseFailAlloc_466_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
return v___x_465_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerParametricAttributeExtra___redArg___boxed(lean_object* v_impl_468_, lean_object* v_extra_469_, lean_object* v_a_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_batteries_Lean_registerParametricAttributeExtra___redArg(v_impl_468_, v_extra_469_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerParametricAttributeExtra(lean_object* v_00_u03b1_472_, lean_object* v_impl_473_, lean_object* v_extra_474_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_batteries_Lean_registerParametricAttributeExtra___redArg(v_impl_473_, v_extra_474_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerParametricAttributeExtra___boxed(lean_object* v_00_u03b1_477_, lean_object* v_impl_478_, lean_object* v_extra_479_, lean_object* v_a_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_batteries_Lean_registerParametricAttributeExtra(v_00_u03b1_477_, v_impl_478_, v_extra_479_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0(lean_object* v_00_u03b2_482_, lean_object* v_m_483_, lean_object* v_a_484_, lean_object* v_b_485_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0___redArg(v_m_483_, v_a_484_, v_b_485_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00Lean_registerParametricAttributeExtra_spec__1(lean_object* v_00_u03b1_487_, lean_object* v_x_488_, lean_object* v_x_489_){
_start:
{
lean_object* v___x_490_; 
v___x_490_ = lp_batteries_List_foldl___at___00Lean_registerParametricAttributeExtra_spec__1___redArg(v_x_488_, v_x_489_);
return v___x_490_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0(lean_object* v_00_u03b2_491_, lean_object* v_a_492_, lean_object* v_x_493_){
_start:
{
uint8_t v___x_494_; 
v___x_494_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___redArg(v_a_492_, v_x_493_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0___boxed(lean_object* v_00_u03b2_495_, lean_object* v_a_496_, lean_object* v_x_497_){
_start:
{
uint8_t v_res_498_; lean_object* v_r_499_; 
v_res_498_ = lp_batteries_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__0(v_00_u03b2_495_, v_a_496_, v_x_497_);
lean_dec(v_x_497_);
lean_dec(v_a_496_);
v_r_499_ = lean_box(v_res_498_);
return v_r_499_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1(lean_object* v_00_u03b2_500_, lean_object* v_data_501_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lp_batteries_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1___redArg(v_data_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__2(lean_object* v_00_u03b2_503_, lean_object* v_a_504_, lean_object* v_b_505_, lean_object* v_x_506_){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = lp_batteries_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__2___redArg(v_a_504_, v_b_505_, v_x_506_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_508_, lean_object* v_i_509_, lean_object* v_source_510_, lean_object* v_target_511_){
_start:
{
lean_object* v___x_512_; 
v___x_512_ = lp_batteries___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2___redArg(v_i_509_, v_source_510_, v_target_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_513_, lean_object* v_x_514_, lean_object* v_x_515_){
_start:
{
lean_object* v___x_516_; 
v___x_516_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Lean_registerParametricAttributeExtra_spec__0_spec__1_spec__2_spec__4___redArg(v_x_514_, v_x_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg(lean_object* v_inst_519_, lean_object* v_attr_520_, lean_object* v_env_521_, lean_object* v_decl_522_){
_start:
{
lean_object* v_attr_523_; lean_object* v_base_524_; lean_object* v___x_525_; 
v_attr_523_ = lean_ctor_get(v_attr_520_, 0);
v_base_524_ = lean_ctor_get(v_attr_520_, 1);
lean_inc(v_decl_522_);
v___x_525_ = l_Lean_ParametricAttribute_getParam_x3f___redArg(v_inst_519_, v_attr_523_, v_env_521_, v_decl_522_);
if (lean_obj_tag(v___x_525_) == 0)
{
lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; 
v___x_526_ = ((lean_object*)(lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___closed__0));
v___x_527_ = ((lean_object*)(lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___closed__1));
v___x_528_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___x_526_, v___x_527_, v_base_524_, v_decl_522_);
return v___x_528_;
}
else
{
lean_dec(v_decl_522_);
return v___x_525_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg___boxed(lean_object* v_inst_529_, lean_object* v_attr_530_, lean_object* v_env_531_, lean_object* v_decl_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg(v_inst_529_, v_attr_530_, v_env_531_, v_decl_532_);
lean_dec_ref(v_attr_530_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f(lean_object* v_00_u03b1_534_, lean_object* v_inst_535_, lean_object* v_attr_536_, lean_object* v_env_537_, lean_object* v_decl_538_){
_start:
{
lean_object* v___x_539_; 
v___x_539_ = lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___redArg(v_inst_535_, v_attr_536_, v_env_537_, v_decl_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f___boxed(lean_object* v_00_u03b1_540_, lean_object* v_inst_541_, lean_object* v_attr_542_, lean_object* v_env_543_, lean_object* v_decl_544_){
_start:
{
lean_object* v_res_545_; 
v_res_545_ = lp_batteries_Lean_ParametricAttributeExtra_getParam_x3f(v_00_u03b1_540_, v_inst_541_, v_attr_542_, v_env_543_, v_decl_544_);
lean_dec_ref(v_attr_542_);
return v_res_545_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_setParam___redArg(lean_object* v_attr_546_, lean_object* v_env_547_, lean_object* v_decl_548_, lean_object* v_param_549_){
_start:
{
lean_object* v_attr_550_; lean_object* v___x_551_; 
v_attr_550_ = lean_ctor_get(v_attr_546_, 0);
lean_inc_ref(v_attr_550_);
lean_dec_ref(v_attr_546_);
v___x_551_ = l_Lean_ParametricAttribute_setParam___redArg(v_attr_550_, v_env_547_, v_decl_548_, v_param_549_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ParametricAttributeExtra_setParam(lean_object* v_00_u03b1_552_, lean_object* v_attr_553_, lean_object* v_env_554_, lean_object* v_decl_555_, lean_object* v_param_556_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = lp_batteries_Lean_ParametricAttributeExtra_setParam___redArg(v_attr_553_, v_env_554_, v_decl_555_, v_param_556_);
return v___x_557_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_TagAttribute(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_HashMap_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_AttributeExtra(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_TagAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_HashMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Lean_instInhabitedTagAttributeExtra_default = _init_lp_batteries_Lean_instInhabitedTagAttributeExtra_default();
lean_mark_persistent(lp_batteries_Lean_instInhabitedTagAttributeExtra_default);
lp_batteries_Lean_instInhabitedTagAttributeExtra = _init_lp_batteries_Lean_instInhabitedTagAttributeExtra();
lean_mark_persistent(lp_batteries_Lean_instInhabitedTagAttributeExtra);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_AttributeExtra(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Lean_registerTagAttributeExtra___auto__1 = _init_lp_batteries_Lean_registerTagAttributeExtra___auto__1();
lean_mark_persistent(lp_batteries_Lean_registerTagAttributeExtra___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_TagAttribute(uint8_t builtin);
lean_object* initialize_Std_Data_HashMap_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_AttributeExtra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_TagAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_HashMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_AttributeExtra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_AttributeExtra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_AttributeExtra(builtin);
}
#ifdef __cplusplus
}
#endif
