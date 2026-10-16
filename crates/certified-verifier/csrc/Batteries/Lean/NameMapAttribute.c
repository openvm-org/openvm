// Lean compiler output
// Module: Batteries.Lean.NameMapAttribute
// Imports: public import Init public meta import Init public import Lean.Attributes
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_thunk_get_own(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_thunk(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_registerSimplePersistentEnvExtension___redArg(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_thunk_pure(lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
extern lean_object* l_Lean_instInhabitedMessageData_default;
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedEnvExtension_default(lean_object*);
lean_object* l_Lean_setEnv___redArg(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "(`Inhabited.default` for `IO.Error`)"};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___closed__0 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 18}, .m_objs = {((lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___closed__0_value)}};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___closed__1 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__1___boxed(lean_object*, lean_object*);
static const lean_array_object lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__0 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__0_value),((lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__0_value),((lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__0_value)}};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__1 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__3___boxed(lean_object*);
static const lean_closure_object lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__0 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__0_value;
static const lean_closure_object lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__1 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__1_value;
static const lean_closure_object lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__2 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__2_value;
static const lean_closure_object lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__3 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__3_value;
static lean_once_cell_t lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__4;
static lean_once_cell_t lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension(lean_object*);
static lean_once_cell_t lp_batteries_Lean_NameMapExtension_find_x3f___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_find_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Already exists entry for "};
static const lean_object* lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__0 = (const lean_object*)&lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1;
static const lean_string_object lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__2 = (const lean_object*)&lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0_value;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__1 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__1_value;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__2 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__2_value;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__3 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__3_value;
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4_value_aux_0),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4_value_aux_1),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4_value_aux_2),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4_value;
static const lean_array_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__5 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__5_value;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__6 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__6_value;
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7_value_aux_0),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7_value_aux_1),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7_value_aux_2),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7_value;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__8 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__8_value;
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__9 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__9_value;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__10 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__10_value;
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11_value_aux_0),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11_value_aux_1),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11_value_aux_2),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11_value;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__12;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__13;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__14 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__14_value;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "declName"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__15 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__15_value;
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16_value_aux_0),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16_value_aux_1),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16_value_aux_2),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(113, 211, 58, 33, 138, 196, 138, 106)}};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16_value;
static const lean_string_object lp_batteries_Lean_registerNameMapExtension___auto__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "decl_name%"};
static const lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__17 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__17_value;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__18;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__19;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__20;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__21;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__22;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__23;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__24;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__25;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__26;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__27;
static lean_once_cell_t lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28;
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___auto__1;
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00Lean_registerNameMapExtension_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__4(lean_object*);
static const lean_closure_object lp_batteries_Lean_registerNameMapExtension___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_registerNameMapExtension___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Lean_registerNameMapExtension___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_registerNameMapExtension___redArg___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___redArg___closed__1_value;
static const lean_closure_object lp_batteries_Lean_registerNameMapExtension___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_registerNameMapExtension___redArg___lam__4, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_registerNameMapExtension___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00Lean_registerNameMapExtension_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapAttributeImpl_ref___autoParam;
static lean_once_cell_t lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__0 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__0_value;
static const lean_string_object lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "instInhabitedNameMapAttributeImpl"};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__1 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__1_value;
static const lean_string_object lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__2 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__2_value;
static const lean_ctor_object lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__3_value_aux_0),((lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__1_value),LEAN_SCALAR_PTR_LITERAL(167, 167, 15, 255, 21, 132, 154, 151)}};
static const lean_ctor_object lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__3_value_aux_1),((lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__2_value),LEAN_SCALAR_PTR_LITERAL(25, 62, 10, 95, 179, 38, 54, 241)}};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__3 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__3_value;
static const lean_string_object lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__4 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__4_value;
static const lean_ctor_object lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__3_value),((lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__4_value),((lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__0_value)}};
static const lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__5 = (const lean_object*)&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default(lean_object*);
static lean_once_cell_t lp_batteries_Lean_instInhabitedNameMapAttributeImpl___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl(lean_object*);
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__0;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__2;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__3;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__4;
static lean_once_cell_t lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__0 = (const lean_object*)&lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__1;
static const lean_string_object lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__2 = (const lean_object*)&lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__1;
static lean_once_cell_t lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_registerNameMapAttribute___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "registerNameMapAttribute"};
static const lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_registerNameMapAttribute___redArg___closed__0_value;
static const lean_ctor_object lp_batteries_Lean_registerNameMapAttribute___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_registerNameMapAttribute___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_registerNameMapAttribute___redArg___closed__1_value_aux_0),((lean_object*)&lp_batteries_Lean_registerNameMapAttribute___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(176, 120, 89, 247, 255, 33, 250, 195)}};
static const lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_registerNameMapAttribute___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0(lean_object* v_x_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = ((lean_object*)(lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___closed__1));
v___x_8_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0___boxed(lean_object* v_x_9_, lean_object* v___y_10_, lean_object* v___y_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__0(v_x_9_, v___y_10_);
lean_dec_ref(v___y_10_);
lean_dec_ref(v_x_9_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__1(lean_object* v_s_13_, lean_object* v_x_14_){
_start:
{
lean_inc_ref(v_s_13_);
return v_s_13_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__1___boxed(lean_object* v_s_15_, lean_object* v_x_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__1(v_s_15_, v_x_16_);
lean_dec_ref(v_x_16_);
lean_dec_ref(v_s_15_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2(lean_object* v_x_22_, lean_object* v_x_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = ((lean_object*)(lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___closed__1));
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2___boxed(lean_object* v_x_25_, lean_object* v_x_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__2(v_x_25_, v_x_26_);
lean_dec_ref(v_x_26_);
lean_dec_ref(v_x_25_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__3(lean_object* v_x_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lean_box(0);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__3___boxed(lean_object* v_x_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___lam__3(v_x_30_);
lean_dec_ref(v_x_30_);
return v_res_31_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__4(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = l_Lean_instInhabitedEnvExtension_default(lean_box(0));
return v___x_36_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5(void){
_start:
{
lean_object* v___f_37_; lean_object* v___f_38_; lean_object* v___f_39_; lean_object* v___f_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___f_37_ = ((lean_object*)(lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__3));
v___f_38_ = ((lean_object*)(lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__2));
v___f_39_ = ((lean_object*)(lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__1));
v___f_40_ = ((lean_object*)(lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__0));
v___x_41_ = lean_box(0);
v___x_42_ = lean_obj_once(&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__4, &lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__4_once, _init_lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__4);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension___aux__1(lean_object* v_00_u03b1_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lean_obj_once(&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5, &lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5_once, _init_lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapExtension(lean_object* v_00_u03b1_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lean_obj_once(&lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5, &lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5_once, _init_lp_batteries_Lean_instInhabitedNameMapExtension___aux__1___closed__5);
return v___x_47_;
}
}
static lean_object* _init_lp_batteries_Lean_NameMapExtension_find_x3f___redArg___closed__0(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = lean_box(1);
v___x_49_ = lean_thunk_pure(v___x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg(lean_object* v_ext_50_, lean_object* v_env_51_, lean_object* v_n_52_){
_start:
{
lean_object* v_toEnvExtension_53_; lean_object* v_asyncMode_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v_toEnvExtension_53_ = lean_ctor_get(v_ext_50_, 0);
v_asyncMode_54_ = lean_ctor_get(v_toEnvExtension_53_, 2);
v___x_55_ = lean_obj_once(&lp_batteries_Lean_NameMapExtension_find_x3f___redArg___closed__0, &lp_batteries_Lean_NameMapExtension_find_x3f___redArg___closed__0_once, _init_lp_batteries_Lean_NameMapExtension_find_x3f___redArg___closed__0);
v___x_56_ = lean_box(0);
v___x_57_ = l_Lean_SimplePersistentEnvExtension_getState___redArg(v___x_55_, v_ext_50_, v_env_51_, v_asyncMode_54_, v___x_56_);
v___x_58_ = lean_thunk_get_own(v___x_57_);
lean_dec(v___x_57_);
v___x_59_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v___x_58_, v_n_52_);
lean_dec(v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___redArg___boxed(lean_object* v_ext_60_, lean_object* v_env_61_, lean_object* v_n_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_60_, v_env_61_, v_n_62_);
lean_dec(v_n_62_);
lean_dec_ref(v_ext_60_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_find_x3f(lean_object* v_00_u03b1_64_, lean_object* v_ext_65_, lean_object* v_env_66_, lean_object* v_n_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_65_, v_env_66_, v_n_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_find_x3f___boxed(lean_object* v_00_u03b1_69_, lean_object* v_ext_70_, lean_object* v_env_71_, lean_object* v_n_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_batteries_Lean_NameMapExtension_find_x3f(v_00_u03b1_69_, v_ext_70_, v_env_71_, v_n_72_);
lean_dec(v_n_72_);
lean_dec_ref(v_ext_70_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___redArg___lam__0(lean_object* v_ext_74_, lean_object* v_k_75_, lean_object* v_v_76_, lean_object* v_inst_77_, lean_object* v_____do__lift_78_){
_start:
{
lean_object* v_toEnvExtension_79_; lean_object* v_asyncMode_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v_toEnvExtension_79_ = lean_ctor_get(v_ext_74_, 0);
v_asyncMode_80_ = lean_ctor_get(v_toEnvExtension_79_, 2);
lean_inc(v_asyncMode_80_);
v___x_81_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_81_, 0, v_k_75_);
lean_ctor_set(v___x_81_, 1, v_v_76_);
v___x_82_ = lean_box(0);
v___x_83_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v_ext_74_, v_____do__lift_78_, v___x_81_, v_asyncMode_80_, v___x_82_);
lean_dec(v_asyncMode_80_);
v___x_84_ = l_Lean_setEnv___redArg(v_inst_77_, v___x_83_);
return v___x_84_;
}
}
static lean_object* _init_lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = ((lean_object*)(lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__0));
v___x_87_ = l_Lean_stringToMessageData(v___x_86_);
return v___x_87_;
}
}
static lean_object* _init_lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_89_ = ((lean_object*)(lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__2));
v___x_90_ = l_Lean_stringToMessageData(v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___redArg___lam__1(lean_object* v_ext_91_, lean_object* v_k_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_toBind_95_, lean_object* v_getEnv_96_, lean_object* v___f_97_, lean_object* v_____do__lift_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_91_, v_____do__lift_98_, v_k_92_);
if (lean_obj_tag(v___x_99_) == 1)
{
lean_object* v_name_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
lean_dec_ref_known(v___x_99_, 1);
lean_dec(v___f_97_);
lean_dec(v_getEnv_96_);
lean_dec(v_toBind_95_);
v_name_100_ = lean_ctor_get(v_ext_91_, 1);
lean_inc(v_name_100_);
lean_dec_ref(v_ext_91_);
v___x_101_ = lean_obj_once(&lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1, &lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1_once, _init_lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1);
v___x_102_ = l_Lean_MessageData_ofName(v_name_100_);
v___x_103_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_101_);
lean_ctor_set(v___x_103_, 1, v___x_102_);
v___x_104_ = lean_obj_once(&lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3, &lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3_once, _init_lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3);
v___x_105_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_103_);
lean_ctor_set(v___x_105_, 1, v___x_104_);
v___x_106_ = l_Lean_MessageData_ofName(v_k_92_);
v___x_107_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_105_);
lean_ctor_set(v___x_107_, 1, v___x_106_);
v___x_108_ = l_Lean_throwError___redArg(v_inst_93_, v_inst_94_, v___x_107_);
return v___x_108_;
}
else
{
lean_object* v___x_109_; 
lean_dec(v___x_99_);
lean_dec_ref(v_inst_94_);
lean_dec_ref(v_inst_93_);
lean_dec(v_k_92_);
lean_dec_ref(v_ext_91_);
v___x_109_ = lean_apply_4(v_toBind_95_, lean_box(0), lean_box(0), v_getEnv_96_, v___f_97_);
return v___x_109_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___redArg(lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_ext_113_, lean_object* v_k_114_, lean_object* v_v_115_){
_start:
{
lean_object* v_toBind_116_; lean_object* v_getEnv_117_; lean_object* v___f_118_; lean_object* v___f_119_; lean_object* v___x_120_; 
v_toBind_116_ = lean_ctor_get(v_inst_110_, 1);
lean_inc_n(v_toBind_116_, 2);
v_getEnv_117_ = lean_ctor_get(v_inst_111_, 0);
lean_inc_n(v_getEnv_117_, 2);
lean_inc(v_k_114_);
lean_inc_ref(v_ext_113_);
v___f_118_ = lean_alloc_closure((void*)(lp_batteries_Lean_NameMapExtension_add___redArg___lam__0), 5, 4);
lean_closure_set(v___f_118_, 0, v_ext_113_);
lean_closure_set(v___f_118_, 1, v_k_114_);
lean_closure_set(v___f_118_, 2, v_v_115_);
lean_closure_set(v___f_118_, 3, v_inst_111_);
v___f_119_ = lean_alloc_closure((void*)(lp_batteries_Lean_NameMapExtension_add___redArg___lam__1), 8, 7);
lean_closure_set(v___f_119_, 0, v_ext_113_);
lean_closure_set(v___f_119_, 1, v_k_114_);
lean_closure_set(v___f_119_, 2, v_inst_110_);
lean_closure_set(v___f_119_, 3, v_inst_112_);
lean_closure_set(v___f_119_, 4, v_toBind_116_);
lean_closure_set(v___f_119_, 5, v_getEnv_117_);
lean_closure_set(v___f_119_, 6, v___f_118_);
v___x_120_ = lean_apply_4(v_toBind_116_, lean_box(0), lean_box(0), v_getEnv_117_, v___f_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add(lean_object* v_M_121_, lean_object* v_00_u03b1_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_ext_126_, lean_object* v_k_127_, lean_object* v_v_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_batteries_Lean_NameMapExtension_add___redArg(v_inst_123_, v_inst_124_, v_inst_125_, v_ext_126_, v_k_127_, v_v_128_);
return v___x_129_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__12(void){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__10));
v___x_157_ = l_Lean_mkAtom(v___x_156_);
return v___x_157_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__13(void){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_158_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__12, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__12_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__12);
v___x_159_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__5));
v___x_160_ = lean_array_push(v___x_159_, v___x_158_);
return v___x_160_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__18(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_169_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__17));
v___x_170_ = l_Lean_mkAtom(v___x_169_);
return v___x_170_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__19(void){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_171_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__18, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__18_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__18);
v___x_172_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__5));
v___x_173_ = lean_array_push(v___x_172_, v___x_171_);
return v___x_173_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__20(void){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_174_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__19, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__19_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__19);
v___x_175_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__16));
v___x_176_ = lean_box(2);
v___x_177_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
lean_ctor_set(v___x_177_, 1, v___x_175_);
lean_ctor_set(v___x_177_, 2, v___x_174_);
return v___x_177_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__21(void){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_178_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__20, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__20_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__20);
v___x_179_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__13, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__13_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__13);
v___x_180_ = lean_array_push(v___x_179_, v___x_178_);
return v___x_180_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__22(void){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_181_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__21, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__21_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__21);
v___x_182_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__11));
v___x_183_ = lean_box(2);
v___x_184_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v___x_182_);
lean_ctor_set(v___x_184_, 2, v___x_181_);
return v___x_184_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__23(void){
_start:
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_185_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__22, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__22_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__22);
v___x_186_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__5));
v___x_187_ = lean_array_push(v___x_186_, v___x_185_);
return v___x_187_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__24(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_188_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__23, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__23_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__23);
v___x_189_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__9));
v___x_190_ = lean_box(2);
v___x_191_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v___x_189_);
lean_ctor_set(v___x_191_, 2, v___x_188_);
return v___x_191_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__25(void){
_start:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_192_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__24, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__24_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__24);
v___x_193_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__5));
v___x_194_ = lean_array_push(v___x_193_, v___x_192_);
return v___x_194_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__26(void){
_start:
{
lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_195_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__25, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__25_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__25);
v___x_196_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__7));
v___x_197_ = lean_box(2);
v___x_198_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_198_, 0, v___x_197_);
lean_ctor_set(v___x_198_, 1, v___x_196_);
lean_ctor_set(v___x_198_, 2, v___x_195_);
return v___x_198_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__27(void){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_199_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__26, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__26_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__26);
v___x_200_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__5));
v___x_201_ = lean_array_push(v___x_200_, v___x_199_);
return v___x_201_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28(void){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_202_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__27, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__27_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__27);
v___x_203_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___auto__1___closed__4));
v___x_204_ = lean_box(2);
v___x_205_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
lean_ctor_set(v___x_205_, 1, v___x_203_);
lean_ctor_set(v___x_205_, 2, v___x_202_);
return v___x_205_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapExtension___auto__1(void){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__0(lean_object* v_n_207_, lean_object* v_s_208_, lean_object* v_x_209_){
_start:
{
lean_object* v_fst_210_; lean_object* v_snd_211_; lean_object* v___x_212_; lean_object* v___x_213_; 
v_fst_210_ = lean_ctor_get(v_n_207_, 0);
lean_inc(v_fst_210_);
v_snd_211_ = lean_ctor_get(v_n_207_, 1);
lean_inc(v_snd_211_);
lean_dec_ref(v_n_207_);
v___x_212_ = lean_thunk_get_own(v_s_208_);
v___x_213_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_fst_210_, v_snd_211_, v___x_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__0___boxed(lean_object* v_n_214_, lean_object* v_s_215_, lean_object* v_x_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_batteries_Lean_registerNameMapExtension___redArg___lam__0(v_n_214_, v_s_215_, v_x_216_);
lean_dec_ref(v_s_215_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__1(lean_object* v_s_218_, lean_object* v_n_219_){
_start:
{
lean_object* v___f_220_; lean_object* v___x_221_; 
v___f_220_ = lean_alloc_closure((void*)(lp_batteries_Lean_registerNameMapExtension___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_220_, 0, v_n_219_);
lean_closure_set(v___f_220_, 1, v_s_218_);
v___x_221_ = lean_mk_thunk(v___f_220_);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__2(lean_object* v_es_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_array_mk(v_es_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00Lean_registerNameMapExtension_spec__0___redArg(lean_object* v_k_224_, lean_object* v_v_225_, lean_object* v_t_226_){
_start:
{
if (lean_obj_tag(v_t_226_) == 0)
{
lean_object* v_size_227_; lean_object* v_k_228_; lean_object* v_v_229_; lean_object* v_l_230_; lean_object* v_r_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_511_; 
v_size_227_ = lean_ctor_get(v_t_226_, 0);
v_k_228_ = lean_ctor_get(v_t_226_, 1);
v_v_229_ = lean_ctor_get(v_t_226_, 2);
v_l_230_ = lean_ctor_get(v_t_226_, 3);
v_r_231_ = lean_ctor_get(v_t_226_, 4);
v_isSharedCheck_511_ = !lean_is_exclusive(v_t_226_);
if (v_isSharedCheck_511_ == 0)
{
v___x_233_ = v_t_226_;
v_isShared_234_ = v_isSharedCheck_511_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_r_231_);
lean_inc(v_l_230_);
lean_inc(v_v_229_);
lean_inc(v_k_228_);
lean_inc(v_size_227_);
lean_dec(v_t_226_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_511_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
uint8_t v___x_235_; 
v___x_235_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_224_, v_k_228_);
switch(v___x_235_)
{
case 0:
{
lean_object* v_impl_236_; lean_object* v___x_237_; 
lean_dec(v_size_227_);
v_impl_236_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00Lean_registerNameMapExtension_spec__0___redArg(v_k_224_, v_v_225_, v_l_230_);
v___x_237_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_231_) == 0)
{
lean_object* v_size_238_; lean_object* v_size_239_; lean_object* v_k_240_; lean_object* v_v_241_; lean_object* v_l_242_; lean_object* v_r_243_; lean_object* v___x_244_; lean_object* v___x_245_; uint8_t v___x_246_; 
v_size_238_ = lean_ctor_get(v_r_231_, 0);
v_size_239_ = lean_ctor_get(v_impl_236_, 0);
lean_inc(v_size_239_);
v_k_240_ = lean_ctor_get(v_impl_236_, 1);
lean_inc(v_k_240_);
v_v_241_ = lean_ctor_get(v_impl_236_, 2);
lean_inc(v_v_241_);
v_l_242_ = lean_ctor_get(v_impl_236_, 3);
lean_inc(v_l_242_);
v_r_243_ = lean_ctor_get(v_impl_236_, 4);
lean_inc(v_r_243_);
v___x_244_ = lean_unsigned_to_nat(3u);
v___x_245_ = lean_nat_mul(v___x_244_, v_size_238_);
v___x_246_ = lean_nat_dec_lt(v___x_245_, v_size_239_);
lean_dec(v___x_245_);
if (v___x_246_ == 0)
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_250_; 
lean_dec(v_r_243_);
lean_dec(v_l_242_);
lean_dec(v_v_241_);
lean_dec(v_k_240_);
v___x_247_ = lean_nat_add(v___x_237_, v_size_239_);
lean_dec(v_size_239_);
v___x_248_ = lean_nat_add(v___x_247_, v_size_238_);
lean_dec(v___x_247_);
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 3, v_impl_236_);
lean_ctor_set(v___x_233_, 0, v___x_248_);
v___x_250_ = v___x_233_;
goto v_reusejp_249_;
}
else
{
lean_object* v_reuseFailAlloc_251_; 
v_reuseFailAlloc_251_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_251_, 0, v___x_248_);
lean_ctor_set(v_reuseFailAlloc_251_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_251_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_251_, 3, v_impl_236_);
lean_ctor_set(v_reuseFailAlloc_251_, 4, v_r_231_);
v___x_250_ = v_reuseFailAlloc_251_;
goto v_reusejp_249_;
}
v_reusejp_249_:
{
return v___x_250_;
}
}
else
{
lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_317_; 
v_isSharedCheck_317_ = !lean_is_exclusive(v_impl_236_);
if (v_isSharedCheck_317_ == 0)
{
lean_object* v_unused_318_; lean_object* v_unused_319_; lean_object* v_unused_320_; lean_object* v_unused_321_; lean_object* v_unused_322_; 
v_unused_318_ = lean_ctor_get(v_impl_236_, 4);
lean_dec(v_unused_318_);
v_unused_319_ = lean_ctor_get(v_impl_236_, 3);
lean_dec(v_unused_319_);
v_unused_320_ = lean_ctor_get(v_impl_236_, 2);
lean_dec(v_unused_320_);
v_unused_321_ = lean_ctor_get(v_impl_236_, 1);
lean_dec(v_unused_321_);
v_unused_322_ = lean_ctor_get(v_impl_236_, 0);
lean_dec(v_unused_322_);
v___x_253_ = v_impl_236_;
v_isShared_254_ = v_isSharedCheck_317_;
goto v_resetjp_252_;
}
else
{
lean_dec(v_impl_236_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_317_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v_size_255_; lean_object* v_size_256_; lean_object* v_k_257_; lean_object* v_v_258_; lean_object* v_l_259_; lean_object* v_r_260_; lean_object* v___x_261_; lean_object* v___x_262_; uint8_t v___x_263_; 
v_size_255_ = lean_ctor_get(v_l_242_, 0);
v_size_256_ = lean_ctor_get(v_r_243_, 0);
v_k_257_ = lean_ctor_get(v_r_243_, 1);
v_v_258_ = lean_ctor_get(v_r_243_, 2);
v_l_259_ = lean_ctor_get(v_r_243_, 3);
v_r_260_ = lean_ctor_get(v_r_243_, 4);
v___x_261_ = lean_unsigned_to_nat(2u);
v___x_262_ = lean_nat_mul(v___x_261_, v_size_255_);
v___x_263_ = lean_nat_dec_lt(v_size_256_, v___x_262_);
lean_dec(v___x_262_);
if (v___x_263_ == 0)
{
lean_object* v___x_265_; uint8_t v_isShared_266_; uint8_t v_isSharedCheck_292_; 
lean_inc(v_r_260_);
lean_inc(v_l_259_);
lean_inc(v_v_258_);
lean_inc(v_k_257_);
v_isSharedCheck_292_ = !lean_is_exclusive(v_r_243_);
if (v_isSharedCheck_292_ == 0)
{
lean_object* v_unused_293_; lean_object* v_unused_294_; lean_object* v_unused_295_; lean_object* v_unused_296_; lean_object* v_unused_297_; 
v_unused_293_ = lean_ctor_get(v_r_243_, 4);
lean_dec(v_unused_293_);
v_unused_294_ = lean_ctor_get(v_r_243_, 3);
lean_dec(v_unused_294_);
v_unused_295_ = lean_ctor_get(v_r_243_, 2);
lean_dec(v_unused_295_);
v_unused_296_ = lean_ctor_get(v_r_243_, 1);
lean_dec(v_unused_296_);
v_unused_297_ = lean_ctor_get(v_r_243_, 0);
lean_dec(v_unused_297_);
v___x_265_ = v_r_243_;
v_isShared_266_ = v_isSharedCheck_292_;
goto v_resetjp_264_;
}
else
{
lean_dec(v_r_243_);
v___x_265_ = lean_box(0);
v_isShared_266_ = v_isSharedCheck_292_;
goto v_resetjp_264_;
}
v_resetjp_264_:
{
lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___y_270_; lean_object* v___y_271_; lean_object* v___y_272_; lean_object* v___x_280_; lean_object* v___y_282_; 
v___x_267_ = lean_nat_add(v___x_237_, v_size_239_);
lean_dec(v_size_239_);
v___x_268_ = lean_nat_add(v___x_267_, v_size_238_);
lean_dec(v___x_267_);
v___x_280_ = lean_nat_add(v___x_237_, v_size_255_);
if (lean_obj_tag(v_l_259_) == 0)
{
lean_object* v_size_290_; 
v_size_290_ = lean_ctor_get(v_l_259_, 0);
lean_inc(v_size_290_);
v___y_282_ = v_size_290_;
goto v___jp_281_;
}
else
{
lean_object* v___x_291_; 
v___x_291_ = lean_unsigned_to_nat(0u);
v___y_282_ = v___x_291_;
goto v___jp_281_;
}
v___jp_269_:
{
lean_object* v___x_273_; lean_object* v___x_275_; 
v___x_273_ = lean_nat_add(v___y_270_, v___y_272_);
lean_dec(v___y_272_);
lean_dec(v___y_270_);
if (v_isShared_266_ == 0)
{
lean_ctor_set(v___x_265_, 4, v_r_231_);
lean_ctor_set(v___x_265_, 3, v_r_260_);
lean_ctor_set(v___x_265_, 2, v_v_229_);
lean_ctor_set(v___x_265_, 1, v_k_228_);
lean_ctor_set(v___x_265_, 0, v___x_273_);
v___x_275_ = v___x_265_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v___x_273_);
lean_ctor_set(v_reuseFailAlloc_279_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_279_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_279_, 3, v_r_260_);
lean_ctor_set(v_reuseFailAlloc_279_, 4, v_r_231_);
v___x_275_ = v_reuseFailAlloc_279_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
lean_object* v___x_277_; 
if (v_isShared_254_ == 0)
{
lean_ctor_set(v___x_253_, 4, v___x_275_);
lean_ctor_set(v___x_253_, 3, v___y_271_);
lean_ctor_set(v___x_253_, 2, v_v_258_);
lean_ctor_set(v___x_253_, 1, v_k_257_);
lean_ctor_set(v___x_253_, 0, v___x_268_);
v___x_277_ = v___x_253_;
goto v_reusejp_276_;
}
else
{
lean_object* v_reuseFailAlloc_278_; 
v_reuseFailAlloc_278_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_278_, 0, v___x_268_);
lean_ctor_set(v_reuseFailAlloc_278_, 1, v_k_257_);
lean_ctor_set(v_reuseFailAlloc_278_, 2, v_v_258_);
lean_ctor_set(v_reuseFailAlloc_278_, 3, v___y_271_);
lean_ctor_set(v_reuseFailAlloc_278_, 4, v___x_275_);
v___x_277_ = v_reuseFailAlloc_278_;
goto v_reusejp_276_;
}
v_reusejp_276_:
{
return v___x_277_;
}
}
}
v___jp_281_:
{
lean_object* v___x_283_; lean_object* v___x_285_; 
v___x_283_ = lean_nat_add(v___x_280_, v___y_282_);
lean_dec(v___y_282_);
lean_dec(v___x_280_);
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v_l_259_);
lean_ctor_set(v___x_233_, 3, v_l_242_);
lean_ctor_set(v___x_233_, 2, v_v_241_);
lean_ctor_set(v___x_233_, 1, v_k_240_);
lean_ctor_set(v___x_233_, 0, v___x_283_);
v___x_285_ = v___x_233_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v___x_283_);
lean_ctor_set(v_reuseFailAlloc_289_, 1, v_k_240_);
lean_ctor_set(v_reuseFailAlloc_289_, 2, v_v_241_);
lean_ctor_set(v_reuseFailAlloc_289_, 3, v_l_242_);
lean_ctor_set(v_reuseFailAlloc_289_, 4, v_l_259_);
v___x_285_ = v_reuseFailAlloc_289_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
lean_object* v___x_286_; 
v___x_286_ = lean_nat_add(v___x_237_, v_size_238_);
if (lean_obj_tag(v_r_260_) == 0)
{
lean_object* v_size_287_; 
v_size_287_ = lean_ctor_get(v_r_260_, 0);
lean_inc(v_size_287_);
v___y_270_ = v___x_286_;
v___y_271_ = v___x_285_;
v___y_272_ = v_size_287_;
goto v___jp_269_;
}
else
{
lean_object* v___x_288_; 
v___x_288_ = lean_unsigned_to_nat(0u);
v___y_270_ = v___x_286_;
v___y_271_ = v___x_285_;
v___y_272_ = v___x_288_;
goto v___jp_269_;
}
}
}
}
}
else
{
lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_303_; 
lean_del_object(v___x_233_);
v___x_298_ = lean_nat_add(v___x_237_, v_size_239_);
lean_dec(v_size_239_);
v___x_299_ = lean_nat_add(v___x_298_, v_size_238_);
lean_dec(v___x_298_);
v___x_300_ = lean_nat_add(v___x_237_, v_size_238_);
v___x_301_ = lean_nat_add(v___x_300_, v_size_256_);
lean_dec(v___x_300_);
lean_inc_ref(v_r_231_);
if (v_isShared_254_ == 0)
{
lean_ctor_set(v___x_253_, 4, v_r_231_);
lean_ctor_set(v___x_253_, 3, v_r_243_);
lean_ctor_set(v___x_253_, 2, v_v_229_);
lean_ctor_set(v___x_253_, 1, v_k_228_);
lean_ctor_set(v___x_253_, 0, v___x_301_);
v___x_303_ = v___x_253_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_316_; 
v_reuseFailAlloc_316_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_316_, 0, v___x_301_);
lean_ctor_set(v_reuseFailAlloc_316_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_316_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_316_, 3, v_r_243_);
lean_ctor_set(v_reuseFailAlloc_316_, 4, v_r_231_);
v___x_303_ = v_reuseFailAlloc_316_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_310_; 
v_isSharedCheck_310_ = !lean_is_exclusive(v_r_231_);
if (v_isSharedCheck_310_ == 0)
{
lean_object* v_unused_311_; lean_object* v_unused_312_; lean_object* v_unused_313_; lean_object* v_unused_314_; lean_object* v_unused_315_; 
v_unused_311_ = lean_ctor_get(v_r_231_, 4);
lean_dec(v_unused_311_);
v_unused_312_ = lean_ctor_get(v_r_231_, 3);
lean_dec(v_unused_312_);
v_unused_313_ = lean_ctor_get(v_r_231_, 2);
lean_dec(v_unused_313_);
v_unused_314_ = lean_ctor_get(v_r_231_, 1);
lean_dec(v_unused_314_);
v_unused_315_ = lean_ctor_get(v_r_231_, 0);
lean_dec(v_unused_315_);
v___x_305_ = v_r_231_;
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
else
{
lean_dec(v_r_231_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_308_; 
if (v_isShared_306_ == 0)
{
lean_ctor_set(v___x_305_, 4, v___x_303_);
lean_ctor_set(v___x_305_, 3, v_l_242_);
lean_ctor_set(v___x_305_, 2, v_v_241_);
lean_ctor_set(v___x_305_, 1, v_k_240_);
lean_ctor_set(v___x_305_, 0, v___x_299_);
v___x_308_ = v___x_305_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v___x_299_);
lean_ctor_set(v_reuseFailAlloc_309_, 1, v_k_240_);
lean_ctor_set(v_reuseFailAlloc_309_, 2, v_v_241_);
lean_ctor_set(v_reuseFailAlloc_309_, 3, v_l_242_);
lean_ctor_set(v_reuseFailAlloc_309_, 4, v___x_303_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_323_; 
v_l_323_ = lean_ctor_get(v_impl_236_, 3);
lean_inc(v_l_323_);
if (lean_obj_tag(v_l_323_) == 0)
{
lean_object* v_r_324_; lean_object* v_k_325_; lean_object* v_v_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_337_; 
v_r_324_ = lean_ctor_get(v_impl_236_, 4);
v_k_325_ = lean_ctor_get(v_impl_236_, 1);
v_v_326_ = lean_ctor_get(v_impl_236_, 2);
v_isSharedCheck_337_ = !lean_is_exclusive(v_impl_236_);
if (v_isSharedCheck_337_ == 0)
{
lean_object* v_unused_338_; lean_object* v_unused_339_; 
v_unused_338_ = lean_ctor_get(v_impl_236_, 3);
lean_dec(v_unused_338_);
v_unused_339_ = lean_ctor_get(v_impl_236_, 0);
lean_dec(v_unused_339_);
v___x_328_ = v_impl_236_;
v_isShared_329_ = v_isSharedCheck_337_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_r_324_);
lean_inc(v_v_326_);
lean_inc(v_k_325_);
lean_dec(v_impl_236_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_337_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___x_330_; lean_object* v___x_332_; 
v___x_330_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_324_);
if (v_isShared_329_ == 0)
{
lean_ctor_set(v___x_328_, 3, v_r_324_);
lean_ctor_set(v___x_328_, 2, v_v_229_);
lean_ctor_set(v___x_328_, 1, v_k_228_);
lean_ctor_set(v___x_328_, 0, v___x_237_);
v___x_332_ = v___x_328_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v___x_237_);
lean_ctor_set(v_reuseFailAlloc_336_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_336_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_336_, 3, v_r_324_);
lean_ctor_set(v_reuseFailAlloc_336_, 4, v_r_324_);
v___x_332_ = v_reuseFailAlloc_336_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
lean_object* v___x_334_; 
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v___x_332_);
lean_ctor_set(v___x_233_, 3, v_l_323_);
lean_ctor_set(v___x_233_, 2, v_v_326_);
lean_ctor_set(v___x_233_, 1, v_k_325_);
lean_ctor_set(v___x_233_, 0, v___x_330_);
v___x_334_ = v___x_233_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v___x_330_);
lean_ctor_set(v_reuseFailAlloc_335_, 1, v_k_325_);
lean_ctor_set(v_reuseFailAlloc_335_, 2, v_v_326_);
lean_ctor_set(v_reuseFailAlloc_335_, 3, v_l_323_);
lean_ctor_set(v_reuseFailAlloc_335_, 4, v___x_332_);
v___x_334_ = v_reuseFailAlloc_335_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
return v___x_334_;
}
}
}
}
else
{
lean_object* v_r_340_; 
v_r_340_ = lean_ctor_get(v_impl_236_, 4);
lean_inc(v_r_340_);
if (lean_obj_tag(v_r_340_) == 0)
{
lean_object* v_k_341_; lean_object* v_v_342_; lean_object* v___x_344_; uint8_t v_isShared_345_; uint8_t v_isSharedCheck_365_; 
v_k_341_ = lean_ctor_get(v_impl_236_, 1);
v_v_342_ = lean_ctor_get(v_impl_236_, 2);
v_isSharedCheck_365_ = !lean_is_exclusive(v_impl_236_);
if (v_isSharedCheck_365_ == 0)
{
lean_object* v_unused_366_; lean_object* v_unused_367_; lean_object* v_unused_368_; 
v_unused_366_ = lean_ctor_get(v_impl_236_, 4);
lean_dec(v_unused_366_);
v_unused_367_ = lean_ctor_get(v_impl_236_, 3);
lean_dec(v_unused_367_);
v_unused_368_ = lean_ctor_get(v_impl_236_, 0);
lean_dec(v_unused_368_);
v___x_344_ = v_impl_236_;
v_isShared_345_ = v_isSharedCheck_365_;
goto v_resetjp_343_;
}
else
{
lean_inc(v_v_342_);
lean_inc(v_k_341_);
lean_dec(v_impl_236_);
v___x_344_ = lean_box(0);
v_isShared_345_ = v_isSharedCheck_365_;
goto v_resetjp_343_;
}
v_resetjp_343_:
{
lean_object* v_k_346_; lean_object* v_v_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_361_; 
v_k_346_ = lean_ctor_get(v_r_340_, 1);
v_v_347_ = lean_ctor_get(v_r_340_, 2);
v_isSharedCheck_361_ = !lean_is_exclusive(v_r_340_);
if (v_isSharedCheck_361_ == 0)
{
lean_object* v_unused_362_; lean_object* v_unused_363_; lean_object* v_unused_364_; 
v_unused_362_ = lean_ctor_get(v_r_340_, 4);
lean_dec(v_unused_362_);
v_unused_363_ = lean_ctor_get(v_r_340_, 3);
lean_dec(v_unused_363_);
v_unused_364_ = lean_ctor_get(v_r_340_, 0);
lean_dec(v_unused_364_);
v___x_349_ = v_r_340_;
v_isShared_350_ = v_isSharedCheck_361_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_v_347_);
lean_inc(v_k_346_);
lean_dec(v_r_340_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_361_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___x_351_; lean_object* v___x_353_; 
v___x_351_ = lean_unsigned_to_nat(3u);
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 4, v_l_323_);
lean_ctor_set(v___x_349_, 3, v_l_323_);
lean_ctor_set(v___x_349_, 2, v_v_342_);
lean_ctor_set(v___x_349_, 1, v_k_341_);
lean_ctor_set(v___x_349_, 0, v___x_237_);
v___x_353_ = v___x_349_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v___x_237_);
lean_ctor_set(v_reuseFailAlloc_360_, 1, v_k_341_);
lean_ctor_set(v_reuseFailAlloc_360_, 2, v_v_342_);
lean_ctor_set(v_reuseFailAlloc_360_, 3, v_l_323_);
lean_ctor_set(v_reuseFailAlloc_360_, 4, v_l_323_);
v___x_353_ = v_reuseFailAlloc_360_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
lean_object* v___x_355_; 
if (v_isShared_345_ == 0)
{
lean_ctor_set(v___x_344_, 4, v_l_323_);
lean_ctor_set(v___x_344_, 2, v_v_229_);
lean_ctor_set(v___x_344_, 1, v_k_228_);
lean_ctor_set(v___x_344_, 0, v___x_237_);
v___x_355_ = v___x_344_;
goto v_reusejp_354_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v___x_237_);
lean_ctor_set(v_reuseFailAlloc_359_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_359_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_359_, 3, v_l_323_);
lean_ctor_set(v_reuseFailAlloc_359_, 4, v_l_323_);
v___x_355_ = v_reuseFailAlloc_359_;
goto v_reusejp_354_;
}
v_reusejp_354_:
{
lean_object* v___x_357_; 
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v___x_355_);
lean_ctor_set(v___x_233_, 3, v___x_353_);
lean_ctor_set(v___x_233_, 2, v_v_347_);
lean_ctor_set(v___x_233_, 1, v_k_346_);
lean_ctor_set(v___x_233_, 0, v___x_351_);
v___x_357_ = v___x_233_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_358_, 0, v___x_351_);
lean_ctor_set(v_reuseFailAlloc_358_, 1, v_k_346_);
lean_ctor_set(v_reuseFailAlloc_358_, 2, v_v_347_);
lean_ctor_set(v_reuseFailAlloc_358_, 3, v___x_353_);
lean_ctor_set(v_reuseFailAlloc_358_, 4, v___x_355_);
v___x_357_ = v_reuseFailAlloc_358_;
goto v_reusejp_356_;
}
v_reusejp_356_:
{
return v___x_357_;
}
}
}
}
}
}
else
{
lean_object* v___x_369_; lean_object* v___x_371_; 
v___x_369_ = lean_unsigned_to_nat(2u);
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v_r_340_);
lean_ctor_set(v___x_233_, 3, v_impl_236_);
lean_ctor_set(v___x_233_, 0, v___x_369_);
v___x_371_ = v___x_233_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v___x_369_);
lean_ctor_set(v_reuseFailAlloc_372_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_372_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_372_, 3, v_impl_236_);
lean_ctor_set(v_reuseFailAlloc_372_, 4, v_r_340_);
v___x_371_ = v_reuseFailAlloc_372_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
return v___x_371_;
}
}
}
}
}
case 1:
{
lean_object* v___x_374_; 
lean_dec(v_v_229_);
lean_dec(v_k_228_);
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 2, v_v_225_);
lean_ctor_set(v___x_233_, 1, v_k_224_);
v___x_374_ = v___x_233_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v_size_227_);
lean_ctor_set(v_reuseFailAlloc_375_, 1, v_k_224_);
lean_ctor_set(v_reuseFailAlloc_375_, 2, v_v_225_);
lean_ctor_set(v_reuseFailAlloc_375_, 3, v_l_230_);
lean_ctor_set(v_reuseFailAlloc_375_, 4, v_r_231_);
v___x_374_ = v_reuseFailAlloc_375_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
return v___x_374_;
}
}
default: 
{
lean_object* v_impl_376_; lean_object* v___x_377_; 
lean_dec(v_size_227_);
v_impl_376_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00Lean_registerNameMapExtension_spec__0___redArg(v_k_224_, v_v_225_, v_r_231_);
v___x_377_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_230_) == 0)
{
lean_object* v_size_378_; lean_object* v_size_379_; lean_object* v_k_380_; lean_object* v_v_381_; lean_object* v_l_382_; lean_object* v_r_383_; lean_object* v___x_384_; lean_object* v___x_385_; uint8_t v___x_386_; 
v_size_378_ = lean_ctor_get(v_l_230_, 0);
v_size_379_ = lean_ctor_get(v_impl_376_, 0);
lean_inc(v_size_379_);
v_k_380_ = lean_ctor_get(v_impl_376_, 1);
lean_inc(v_k_380_);
v_v_381_ = lean_ctor_get(v_impl_376_, 2);
lean_inc(v_v_381_);
v_l_382_ = lean_ctor_get(v_impl_376_, 3);
lean_inc(v_l_382_);
v_r_383_ = lean_ctor_get(v_impl_376_, 4);
lean_inc(v_r_383_);
v___x_384_ = lean_unsigned_to_nat(3u);
v___x_385_ = lean_nat_mul(v___x_384_, v_size_378_);
v___x_386_ = lean_nat_dec_lt(v___x_385_, v_size_379_);
lean_dec(v___x_385_);
if (v___x_386_ == 0)
{
lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_390_; 
lean_dec(v_r_383_);
lean_dec(v_l_382_);
lean_dec(v_v_381_);
lean_dec(v_k_380_);
v___x_387_ = lean_nat_add(v___x_377_, v_size_378_);
v___x_388_ = lean_nat_add(v___x_387_, v_size_379_);
lean_dec(v_size_379_);
lean_dec(v___x_387_);
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v_impl_376_);
lean_ctor_set(v___x_233_, 0, v___x_388_);
v___x_390_ = v___x_233_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v___x_388_);
lean_ctor_set(v_reuseFailAlloc_391_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_391_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_391_, 3, v_l_230_);
lean_ctor_set(v_reuseFailAlloc_391_, 4, v_impl_376_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
return v___x_390_;
}
}
else
{
lean_object* v___x_393_; uint8_t v_isShared_394_; uint8_t v_isSharedCheck_455_; 
v_isSharedCheck_455_ = !lean_is_exclusive(v_impl_376_);
if (v_isSharedCheck_455_ == 0)
{
lean_object* v_unused_456_; lean_object* v_unused_457_; lean_object* v_unused_458_; lean_object* v_unused_459_; lean_object* v_unused_460_; 
v_unused_456_ = lean_ctor_get(v_impl_376_, 4);
lean_dec(v_unused_456_);
v_unused_457_ = lean_ctor_get(v_impl_376_, 3);
lean_dec(v_unused_457_);
v_unused_458_ = lean_ctor_get(v_impl_376_, 2);
lean_dec(v_unused_458_);
v_unused_459_ = lean_ctor_get(v_impl_376_, 1);
lean_dec(v_unused_459_);
v_unused_460_ = lean_ctor_get(v_impl_376_, 0);
lean_dec(v_unused_460_);
v___x_393_ = v_impl_376_;
v_isShared_394_ = v_isSharedCheck_455_;
goto v_resetjp_392_;
}
else
{
lean_dec(v_impl_376_);
v___x_393_ = lean_box(0);
v_isShared_394_ = v_isSharedCheck_455_;
goto v_resetjp_392_;
}
v_resetjp_392_:
{
lean_object* v_size_395_; lean_object* v_k_396_; lean_object* v_v_397_; lean_object* v_l_398_; lean_object* v_r_399_; lean_object* v_size_400_; lean_object* v___x_401_; lean_object* v___x_402_; uint8_t v___x_403_; 
v_size_395_ = lean_ctor_get(v_l_382_, 0);
v_k_396_ = lean_ctor_get(v_l_382_, 1);
v_v_397_ = lean_ctor_get(v_l_382_, 2);
v_l_398_ = lean_ctor_get(v_l_382_, 3);
v_r_399_ = lean_ctor_get(v_l_382_, 4);
v_size_400_ = lean_ctor_get(v_r_383_, 0);
v___x_401_ = lean_unsigned_to_nat(2u);
v___x_402_ = lean_nat_mul(v___x_401_, v_size_400_);
v___x_403_ = lean_nat_dec_lt(v_size_395_, v___x_402_);
lean_dec(v___x_402_);
if (v___x_403_ == 0)
{
lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_431_; 
lean_inc(v_r_399_);
lean_inc(v_l_398_);
lean_inc(v_v_397_);
lean_inc(v_k_396_);
v_isSharedCheck_431_ = !lean_is_exclusive(v_l_382_);
if (v_isSharedCheck_431_ == 0)
{
lean_object* v_unused_432_; lean_object* v_unused_433_; lean_object* v_unused_434_; lean_object* v_unused_435_; lean_object* v_unused_436_; 
v_unused_432_ = lean_ctor_get(v_l_382_, 4);
lean_dec(v_unused_432_);
v_unused_433_ = lean_ctor_get(v_l_382_, 3);
lean_dec(v_unused_433_);
v_unused_434_ = lean_ctor_get(v_l_382_, 2);
lean_dec(v_unused_434_);
v_unused_435_ = lean_ctor_get(v_l_382_, 1);
lean_dec(v_unused_435_);
v_unused_436_ = lean_ctor_get(v_l_382_, 0);
lean_dec(v_unused_436_);
v___x_405_ = v_l_382_;
v_isShared_406_ = v_isSharedCheck_431_;
goto v_resetjp_404_;
}
else
{
lean_dec(v_l_382_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_431_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___y_410_; lean_object* v___y_411_; lean_object* v___y_412_; lean_object* v___y_421_; 
v___x_407_ = lean_nat_add(v___x_377_, v_size_378_);
v___x_408_ = lean_nat_add(v___x_407_, v_size_379_);
lean_dec(v_size_379_);
if (lean_obj_tag(v_l_398_) == 0)
{
lean_object* v_size_429_; 
v_size_429_ = lean_ctor_get(v_l_398_, 0);
lean_inc(v_size_429_);
v___y_421_ = v_size_429_;
goto v___jp_420_;
}
else
{
lean_object* v___x_430_; 
v___x_430_ = lean_unsigned_to_nat(0u);
v___y_421_ = v___x_430_;
goto v___jp_420_;
}
v___jp_409_:
{
lean_object* v___x_413_; lean_object* v___x_415_; 
v___x_413_ = lean_nat_add(v___y_410_, v___y_412_);
lean_dec(v___y_412_);
lean_dec(v___y_410_);
if (v_isShared_406_ == 0)
{
lean_ctor_set(v___x_405_, 4, v_r_383_);
lean_ctor_set(v___x_405_, 3, v_r_399_);
lean_ctor_set(v___x_405_, 2, v_v_381_);
lean_ctor_set(v___x_405_, 1, v_k_380_);
lean_ctor_set(v___x_405_, 0, v___x_413_);
v___x_415_ = v___x_405_;
goto v_reusejp_414_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v___x_413_);
lean_ctor_set(v_reuseFailAlloc_419_, 1, v_k_380_);
lean_ctor_set(v_reuseFailAlloc_419_, 2, v_v_381_);
lean_ctor_set(v_reuseFailAlloc_419_, 3, v_r_399_);
lean_ctor_set(v_reuseFailAlloc_419_, 4, v_r_383_);
v___x_415_ = v_reuseFailAlloc_419_;
goto v_reusejp_414_;
}
v_reusejp_414_:
{
lean_object* v___x_417_; 
if (v_isShared_394_ == 0)
{
lean_ctor_set(v___x_393_, 4, v___x_415_);
lean_ctor_set(v___x_393_, 3, v___y_411_);
lean_ctor_set(v___x_393_, 2, v_v_397_);
lean_ctor_set(v___x_393_, 1, v_k_396_);
lean_ctor_set(v___x_393_, 0, v___x_408_);
v___x_417_ = v___x_393_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_418_; 
v_reuseFailAlloc_418_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_418_, 0, v___x_408_);
lean_ctor_set(v_reuseFailAlloc_418_, 1, v_k_396_);
lean_ctor_set(v_reuseFailAlloc_418_, 2, v_v_397_);
lean_ctor_set(v_reuseFailAlloc_418_, 3, v___y_411_);
lean_ctor_set(v_reuseFailAlloc_418_, 4, v___x_415_);
v___x_417_ = v_reuseFailAlloc_418_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
return v___x_417_;
}
}
}
v___jp_420_:
{
lean_object* v___x_422_; lean_object* v___x_424_; 
v___x_422_ = lean_nat_add(v___x_407_, v___y_421_);
lean_dec(v___y_421_);
lean_dec(v___x_407_);
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v_l_398_);
lean_ctor_set(v___x_233_, 0, v___x_422_);
v___x_424_ = v___x_233_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_422_);
lean_ctor_set(v_reuseFailAlloc_428_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_428_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_428_, 3, v_l_230_);
lean_ctor_set(v_reuseFailAlloc_428_, 4, v_l_398_);
v___x_424_ = v_reuseFailAlloc_428_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
lean_object* v___x_425_; 
v___x_425_ = lean_nat_add(v___x_377_, v_size_400_);
if (lean_obj_tag(v_r_399_) == 0)
{
lean_object* v_size_426_; 
v_size_426_ = lean_ctor_get(v_r_399_, 0);
lean_inc(v_size_426_);
v___y_410_ = v___x_425_;
v___y_411_ = v___x_424_;
v___y_412_ = v_size_426_;
goto v___jp_409_;
}
else
{
lean_object* v___x_427_; 
v___x_427_ = lean_unsigned_to_nat(0u);
v___y_410_ = v___x_425_;
v___y_411_ = v___x_424_;
v___y_412_ = v___x_427_;
goto v___jp_409_;
}
}
}
}
}
else
{
lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_441_; 
lean_del_object(v___x_233_);
v___x_437_ = lean_nat_add(v___x_377_, v_size_378_);
v___x_438_ = lean_nat_add(v___x_437_, v_size_379_);
lean_dec(v_size_379_);
v___x_439_ = lean_nat_add(v___x_437_, v_size_395_);
lean_dec(v___x_437_);
lean_inc_ref(v_l_230_);
if (v_isShared_394_ == 0)
{
lean_ctor_set(v___x_393_, 4, v_l_382_);
lean_ctor_set(v___x_393_, 3, v_l_230_);
lean_ctor_set(v___x_393_, 2, v_v_229_);
lean_ctor_set(v___x_393_, 1, v_k_228_);
lean_ctor_set(v___x_393_, 0, v___x_439_);
v___x_441_ = v___x_393_;
goto v_reusejp_440_;
}
else
{
lean_object* v_reuseFailAlloc_454_; 
v_reuseFailAlloc_454_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_454_, 0, v___x_439_);
lean_ctor_set(v_reuseFailAlloc_454_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_454_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_454_, 3, v_l_230_);
lean_ctor_set(v_reuseFailAlloc_454_, 4, v_l_382_);
v___x_441_ = v_reuseFailAlloc_454_;
goto v_reusejp_440_;
}
v_reusejp_440_:
{
lean_object* v___x_443_; uint8_t v_isShared_444_; uint8_t v_isSharedCheck_448_; 
v_isSharedCheck_448_ = !lean_is_exclusive(v_l_230_);
if (v_isSharedCheck_448_ == 0)
{
lean_object* v_unused_449_; lean_object* v_unused_450_; lean_object* v_unused_451_; lean_object* v_unused_452_; lean_object* v_unused_453_; 
v_unused_449_ = lean_ctor_get(v_l_230_, 4);
lean_dec(v_unused_449_);
v_unused_450_ = lean_ctor_get(v_l_230_, 3);
lean_dec(v_unused_450_);
v_unused_451_ = lean_ctor_get(v_l_230_, 2);
lean_dec(v_unused_451_);
v_unused_452_ = lean_ctor_get(v_l_230_, 1);
lean_dec(v_unused_452_);
v_unused_453_ = lean_ctor_get(v_l_230_, 0);
lean_dec(v_unused_453_);
v___x_443_ = v_l_230_;
v_isShared_444_ = v_isSharedCheck_448_;
goto v_resetjp_442_;
}
else
{
lean_dec(v_l_230_);
v___x_443_ = lean_box(0);
v_isShared_444_ = v_isSharedCheck_448_;
goto v_resetjp_442_;
}
v_resetjp_442_:
{
lean_object* v___x_446_; 
if (v_isShared_444_ == 0)
{
lean_ctor_set(v___x_443_, 4, v_r_383_);
lean_ctor_set(v___x_443_, 3, v___x_441_);
lean_ctor_set(v___x_443_, 2, v_v_381_);
lean_ctor_set(v___x_443_, 1, v_k_380_);
lean_ctor_set(v___x_443_, 0, v___x_438_);
v___x_446_ = v___x_443_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_447_; 
v_reuseFailAlloc_447_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_447_, 0, v___x_438_);
lean_ctor_set(v_reuseFailAlloc_447_, 1, v_k_380_);
lean_ctor_set(v_reuseFailAlloc_447_, 2, v_v_381_);
lean_ctor_set(v_reuseFailAlloc_447_, 3, v___x_441_);
lean_ctor_set(v_reuseFailAlloc_447_, 4, v_r_383_);
v___x_446_ = v_reuseFailAlloc_447_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
return v___x_446_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_461_; 
v_l_461_ = lean_ctor_get(v_impl_376_, 3);
lean_inc(v_l_461_);
if (lean_obj_tag(v_l_461_) == 0)
{
lean_object* v_r_462_; lean_object* v_k_463_; lean_object* v_v_464_; lean_object* v___x_466_; uint8_t v_isShared_467_; uint8_t v_isSharedCheck_487_; 
v_r_462_ = lean_ctor_get(v_impl_376_, 4);
v_k_463_ = lean_ctor_get(v_impl_376_, 1);
v_v_464_ = lean_ctor_get(v_impl_376_, 2);
v_isSharedCheck_487_ = !lean_is_exclusive(v_impl_376_);
if (v_isSharedCheck_487_ == 0)
{
lean_object* v_unused_488_; lean_object* v_unused_489_; 
v_unused_488_ = lean_ctor_get(v_impl_376_, 3);
lean_dec(v_unused_488_);
v_unused_489_ = lean_ctor_get(v_impl_376_, 0);
lean_dec(v_unused_489_);
v___x_466_ = v_impl_376_;
v_isShared_467_ = v_isSharedCheck_487_;
goto v_resetjp_465_;
}
else
{
lean_inc(v_r_462_);
lean_inc(v_v_464_);
lean_inc(v_k_463_);
lean_dec(v_impl_376_);
v___x_466_ = lean_box(0);
v_isShared_467_ = v_isSharedCheck_487_;
goto v_resetjp_465_;
}
v_resetjp_465_:
{
lean_object* v_k_468_; lean_object* v_v_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_483_; 
v_k_468_ = lean_ctor_get(v_l_461_, 1);
v_v_469_ = lean_ctor_get(v_l_461_, 2);
v_isSharedCheck_483_ = !lean_is_exclusive(v_l_461_);
if (v_isSharedCheck_483_ == 0)
{
lean_object* v_unused_484_; lean_object* v_unused_485_; lean_object* v_unused_486_; 
v_unused_484_ = lean_ctor_get(v_l_461_, 4);
lean_dec(v_unused_484_);
v_unused_485_ = lean_ctor_get(v_l_461_, 3);
lean_dec(v_unused_485_);
v_unused_486_ = lean_ctor_get(v_l_461_, 0);
lean_dec(v_unused_486_);
v___x_471_ = v_l_461_;
v_isShared_472_ = v_isSharedCheck_483_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_v_469_);
lean_inc(v_k_468_);
lean_dec(v_l_461_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_483_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
lean_object* v___x_473_; lean_object* v___x_475_; 
v___x_473_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_462_, 2);
if (v_isShared_472_ == 0)
{
lean_ctor_set(v___x_471_, 4, v_r_462_);
lean_ctor_set(v___x_471_, 3, v_r_462_);
lean_ctor_set(v___x_471_, 2, v_v_229_);
lean_ctor_set(v___x_471_, 1, v_k_228_);
lean_ctor_set(v___x_471_, 0, v___x_377_);
v___x_475_ = v___x_471_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_482_; 
v_reuseFailAlloc_482_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_482_, 0, v___x_377_);
lean_ctor_set(v_reuseFailAlloc_482_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_482_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_482_, 3, v_r_462_);
lean_ctor_set(v_reuseFailAlloc_482_, 4, v_r_462_);
v___x_475_ = v_reuseFailAlloc_482_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
lean_object* v___x_477_; 
lean_inc(v_r_462_);
if (v_isShared_467_ == 0)
{
lean_ctor_set(v___x_466_, 3, v_r_462_);
lean_ctor_set(v___x_466_, 0, v___x_377_);
v___x_477_ = v___x_466_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v___x_377_);
lean_ctor_set(v_reuseFailAlloc_481_, 1, v_k_463_);
lean_ctor_set(v_reuseFailAlloc_481_, 2, v_v_464_);
lean_ctor_set(v_reuseFailAlloc_481_, 3, v_r_462_);
lean_ctor_set(v_reuseFailAlloc_481_, 4, v_r_462_);
v___x_477_ = v_reuseFailAlloc_481_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
lean_object* v___x_479_; 
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v___x_477_);
lean_ctor_set(v___x_233_, 3, v___x_475_);
lean_ctor_set(v___x_233_, 2, v_v_469_);
lean_ctor_set(v___x_233_, 1, v_k_468_);
lean_ctor_set(v___x_233_, 0, v___x_473_);
v___x_479_ = v___x_233_;
goto v_reusejp_478_;
}
else
{
lean_object* v_reuseFailAlloc_480_; 
v_reuseFailAlloc_480_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_480_, 0, v___x_473_);
lean_ctor_set(v_reuseFailAlloc_480_, 1, v_k_468_);
lean_ctor_set(v_reuseFailAlloc_480_, 2, v_v_469_);
lean_ctor_set(v_reuseFailAlloc_480_, 3, v___x_475_);
lean_ctor_set(v_reuseFailAlloc_480_, 4, v___x_477_);
v___x_479_ = v_reuseFailAlloc_480_;
goto v_reusejp_478_;
}
v_reusejp_478_:
{
return v___x_479_;
}
}
}
}
}
}
else
{
lean_object* v_r_490_; 
v_r_490_ = lean_ctor_get(v_impl_376_, 4);
lean_inc(v_r_490_);
if (lean_obj_tag(v_r_490_) == 0)
{
lean_object* v_k_491_; lean_object* v_v_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_503_; 
v_k_491_ = lean_ctor_get(v_impl_376_, 1);
v_v_492_ = lean_ctor_get(v_impl_376_, 2);
v_isSharedCheck_503_ = !lean_is_exclusive(v_impl_376_);
if (v_isSharedCheck_503_ == 0)
{
lean_object* v_unused_504_; lean_object* v_unused_505_; lean_object* v_unused_506_; 
v_unused_504_ = lean_ctor_get(v_impl_376_, 4);
lean_dec(v_unused_504_);
v_unused_505_ = lean_ctor_get(v_impl_376_, 3);
lean_dec(v_unused_505_);
v_unused_506_ = lean_ctor_get(v_impl_376_, 0);
lean_dec(v_unused_506_);
v___x_494_ = v_impl_376_;
v_isShared_495_ = v_isSharedCheck_503_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_v_492_);
lean_inc(v_k_491_);
lean_dec(v_impl_376_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_503_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_496_; lean_object* v___x_498_; 
v___x_496_ = lean_unsigned_to_nat(3u);
if (v_isShared_495_ == 0)
{
lean_ctor_set(v___x_494_, 4, v_l_461_);
lean_ctor_set(v___x_494_, 2, v_v_229_);
lean_ctor_set(v___x_494_, 1, v_k_228_);
lean_ctor_set(v___x_494_, 0, v___x_377_);
v___x_498_ = v___x_494_;
goto v_reusejp_497_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v___x_377_);
lean_ctor_set(v_reuseFailAlloc_502_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_502_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_502_, 3, v_l_461_);
lean_ctor_set(v_reuseFailAlloc_502_, 4, v_l_461_);
v___x_498_ = v_reuseFailAlloc_502_;
goto v_reusejp_497_;
}
v_reusejp_497_:
{
lean_object* v___x_500_; 
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v_r_490_);
lean_ctor_set(v___x_233_, 3, v___x_498_);
lean_ctor_set(v___x_233_, 2, v_v_492_);
lean_ctor_set(v___x_233_, 1, v_k_491_);
lean_ctor_set(v___x_233_, 0, v___x_496_);
v___x_500_ = v___x_233_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v___x_496_);
lean_ctor_set(v_reuseFailAlloc_501_, 1, v_k_491_);
lean_ctor_set(v_reuseFailAlloc_501_, 2, v_v_492_);
lean_ctor_set(v_reuseFailAlloc_501_, 3, v___x_498_);
lean_ctor_set(v_reuseFailAlloc_501_, 4, v_r_490_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
}
}
else
{
lean_object* v___x_507_; lean_object* v___x_509_; 
v___x_507_ = lean_unsigned_to_nat(2u);
if (v_isShared_234_ == 0)
{
lean_ctor_set(v___x_233_, 4, v_impl_376_);
lean_ctor_set(v___x_233_, 3, v_r_490_);
lean_ctor_set(v___x_233_, 0, v___x_507_);
v___x_509_ = v___x_233_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_510_; 
v_reuseFailAlloc_510_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_510_, 0, v___x_507_);
lean_ctor_set(v_reuseFailAlloc_510_, 1, v_k_228_);
lean_ctor_set(v_reuseFailAlloc_510_, 2, v_v_229_);
lean_ctor_set(v_reuseFailAlloc_510_, 3, v_r_490_);
lean_ctor_set(v_reuseFailAlloc_510_, 4, v_impl_376_);
v___x_509_ = v_reuseFailAlloc_510_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
return v___x_509_;
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
lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_512_ = lean_unsigned_to_nat(1u);
v___x_513_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_513_, 0, v___x_512_);
lean_ctor_set(v___x_513_, 1, v_k_224_);
lean_ctor_set(v___x_513_, 2, v_v_225_);
lean_ctor_set(v___x_513_, 3, v_t_226_);
lean_ctor_set(v___x_513_, 4, v_t_226_);
return v___x_513_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___redArg(lean_object* v_as_514_, size_t v_sz_515_, size_t v_i_516_, lean_object* v_b_517_){
_start:
{
uint8_t v___x_518_; 
v___x_518_ = lean_usize_dec_lt(v_i_516_, v_sz_515_);
if (v___x_518_ == 0)
{
return v_b_517_;
}
else
{
lean_object* v_a_519_; lean_object* v_fst_520_; lean_object* v_snd_521_; lean_object* v_r_522_; size_t v___x_523_; size_t v___x_524_; 
v_a_519_ = lean_array_uget_borrowed(v_as_514_, v_i_516_);
v_fst_520_ = lean_ctor_get(v_a_519_, 0);
v_snd_521_ = lean_ctor_get(v_a_519_, 1);
lean_inc(v_snd_521_);
lean_inc(v_fst_520_);
v_r_522_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00Lean_registerNameMapExtension_spec__0___redArg(v_fst_520_, v_snd_521_, v_b_517_);
v___x_523_ = ((size_t)1ULL);
v___x_524_ = lean_usize_add(v_i_516_, v___x_523_);
v_i_516_ = v___x_524_;
v_b_517_ = v_r_522_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___redArg___boxed(lean_object* v_as_526_, lean_object* v_sz_527_, lean_object* v_i_528_, lean_object* v_b_529_){
_start:
{
size_t v_sz_boxed_530_; size_t v_i_boxed_531_; lean_object* v_res_532_; 
v_sz_boxed_530_ = lean_unbox_usize(v_sz_527_);
lean_dec(v_sz_527_);
v_i_boxed_531_ = lean_unbox_usize(v_i_528_);
lean_dec(v_i_528_);
v_res_532_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___redArg(v_as_526_, v_sz_boxed_530_, v_i_boxed_531_, v_b_529_);
lean_dec_ref(v_as_526_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___redArg(lean_object* v_as_533_, size_t v_i_534_, size_t v_stop_535_, lean_object* v_b_536_){
_start:
{
uint8_t v___x_537_; 
v___x_537_ = lean_usize_dec_eq(v_i_534_, v_stop_535_);
if (v___x_537_ == 0)
{
lean_object* v___x_538_; size_t v_sz_539_; size_t v___x_540_; lean_object* v___x_541_; size_t v___x_542_; size_t v___x_543_; 
v___x_538_ = lean_array_uget_borrowed(v_as_533_, v_i_534_);
v_sz_539_ = lean_array_size(v___x_538_);
v___x_540_ = ((size_t)0ULL);
v___x_541_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___redArg(v___x_538_, v_sz_539_, v___x_540_, v_b_536_);
v___x_542_ = ((size_t)1ULL);
v___x_543_ = lean_usize_add(v_i_534_, v___x_542_);
v_i_534_ = v___x_543_;
v_b_536_ = v___x_541_;
goto _start;
}
else
{
return v_b_536_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___redArg___boxed(lean_object* v_as_545_, lean_object* v_i_546_, lean_object* v_stop_547_, lean_object* v_b_548_){
_start:
{
size_t v_i_boxed_549_; size_t v_stop_boxed_550_; lean_object* v_res_551_; 
v_i_boxed_549_ = lean_unbox_usize(v_i_546_);
lean_dec(v_i_546_);
v_stop_boxed_550_ = lean_unbox_usize(v_stop_547_);
lean_dec(v_stop_547_);
v_res_551_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___redArg(v_as_545_, v_i_boxed_549_, v_stop_boxed_550_, v_b_548_);
lean_dec_ref(v_as_545_);
return v_res_551_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__3(lean_object* v_arr_552_, lean_object* v_x_553_){
_start:
{
lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; uint8_t v___x_557_; 
v___x_554_ = lean_box(1);
v___x_555_ = lean_unsigned_to_nat(0u);
v___x_556_ = lean_array_get_size(v_arr_552_);
v___x_557_ = lean_nat_dec_lt(v___x_555_, v___x_556_);
if (v___x_557_ == 0)
{
return v___x_554_;
}
else
{
uint8_t v___x_558_; 
v___x_558_ = lean_nat_dec_le(v___x_556_, v___x_556_);
if (v___x_558_ == 0)
{
if (v___x_557_ == 0)
{
return v___x_554_;
}
else
{
size_t v___x_559_; size_t v___x_560_; lean_object* v___x_561_; 
v___x_559_ = ((size_t)0ULL);
v___x_560_ = lean_usize_of_nat(v___x_556_);
v___x_561_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___redArg(v_arr_552_, v___x_559_, v___x_560_, v___x_554_);
return v___x_561_;
}
}
else
{
size_t v___x_562_; size_t v___x_563_; lean_object* v___x_564_; 
v___x_562_ = ((size_t)0ULL);
v___x_563_ = lean_usize_of_nat(v___x_556_);
v___x_564_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___redArg(v_arr_552_, v___x_562_, v___x_563_, v___x_554_);
return v___x_564_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__3___boxed(lean_object* v_arr_565_, lean_object* v_x_566_){
_start:
{
lean_object* v_res_567_; 
v_res_567_ = lp_batteries_Lean_registerNameMapExtension___redArg___lam__3(v_arr_565_, v_x_566_);
lean_dec_ref(v_arr_565_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___lam__4(lean_object* v_arr_568_){
_start:
{
lean_object* v___f_569_; lean_object* v___x_570_; 
v___f_569_ = lean_alloc_closure((void*)(lp_batteries_Lean_registerNameMapExtension___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_569_, 0, v_arr_568_);
v___x_570_ = lean_mk_thunk(v___f_569_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg(lean_object* v_name_574_){
_start:
{
lean_object* v___f_576_; lean_object* v___f_577_; lean_object* v___f_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
v___f_576_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___redArg___closed__0));
v___f_577_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___redArg___closed__1));
v___f_578_ = ((lean_object*)(lp_batteries_Lean_registerNameMapExtension___redArg___closed__2));
v___x_579_ = lean_box(0);
v___x_580_ = lean_box(2);
v___x_581_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_581_, 0, v_name_574_);
lean_ctor_set(v___x_581_, 1, v___f_576_);
lean_ctor_set(v___x_581_, 2, v___f_578_);
lean_ctor_set(v___x_581_, 3, v___f_577_);
lean_ctor_set(v___x_581_, 4, v___x_579_);
lean_ctor_set(v___x_581_, 5, v___x_580_);
lean_ctor_set(v___x_581_, 6, v___x_579_);
v___x_582_ = l_Lean_registerSimplePersistentEnvExtension___redArg(v___x_581_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___redArg___boxed(lean_object* v_name_583_, lean_object* v_a_584_){
_start:
{
lean_object* v_res_585_; 
v_res_585_ = lp_batteries_Lean_registerNameMapExtension___redArg(v_name_583_);
return v_res_585_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension(lean_object* v_00_u03b1_586_, lean_object* v_name_587_){
_start:
{
lean_object* v___x_589_; 
v___x_589_ = lp_batteries_Lean_registerNameMapExtension___redArg(v_name_587_);
return v___x_589_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapExtension___boxed(lean_object* v_00_u03b1_590_, lean_object* v_name_591_, lean_object* v_a_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_batteries_Lean_registerNameMapExtension(v_00_u03b1_590_, v_name_591_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00Lean_registerNameMapExtension_spec__0(lean_object* v_00_u03b2_594_, lean_object* v_k_595_, lean_object* v_v_596_, lean_object* v_t_597_, lean_object* v_hl_598_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lp_batteries_Std_DTreeMap_Internal_Impl_insert___at___00Lean_registerNameMapExtension_spec__0___redArg(v_k_595_, v_v_596_, v_t_597_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1(lean_object* v_00_u03b1_600_, lean_object* v_as_601_, size_t v_sz_602_, size_t v_i_603_, lean_object* v_b_604_){
_start:
{
lean_object* v___x_605_; 
v___x_605_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___redArg(v_as_601_, v_sz_602_, v_i_603_, v_b_604_);
return v___x_605_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1___boxed(lean_object* v_00_u03b1_606_, lean_object* v_as_607_, lean_object* v_sz_608_, lean_object* v_i_609_, lean_object* v_b_610_){
_start:
{
size_t v_sz_boxed_611_; size_t v_i_boxed_612_; lean_object* v_res_613_; 
v_sz_boxed_611_ = lean_unbox_usize(v_sz_608_);
lean_dec(v_sz_608_);
v_i_boxed_612_ = lean_unbox_usize(v_i_609_);
lean_dec(v_i_609_);
v_res_613_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_registerNameMapExtension_spec__1(v_00_u03b1_606_, v_as_607_, v_sz_boxed_611_, v_i_boxed_612_, v_b_610_);
lean_dec_ref(v_as_607_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2(lean_object* v_00_u03b1_614_, lean_object* v_as_615_, size_t v_i_616_, size_t v_stop_617_, lean_object* v_b_618_){
_start:
{
lean_object* v___x_619_; 
v___x_619_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___redArg(v_as_615_, v_i_616_, v_stop_617_, v_b_618_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2___boxed(lean_object* v_00_u03b1_620_, lean_object* v_as_621_, lean_object* v_i_622_, lean_object* v_stop_623_, lean_object* v_b_624_){
_start:
{
size_t v_i_boxed_625_; size_t v_stop_boxed_626_; lean_object* v_res_627_; 
v_i_boxed_625_ = lean_unbox_usize(v_i_622_);
lean_dec(v_i_622_);
v_stop_boxed_626_ = lean_unbox_usize(v_stop_623_);
lean_dec(v_stop_623_);
v_res_627_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_registerNameMapExtension_spec__2(v_00_u03b1_620_, v_as_621_, v_i_boxed_625_, v_stop_boxed_626_, v_b_624_);
lean_dec_ref(v_as_621_);
return v_res_627_;
}
}
static lean_object* _init_lp_batteries_Lean_NameMapAttributeImpl_ref___autoParam(void){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = lean_obj_once(&lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28, &lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28_once, _init_lp_batteries_Lean_registerNameMapExtension___auto__1___closed__28);
return v___x_628_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___closed__0(void){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_629_ = l_Lean_instInhabitedMessageData_default;
v___x_630_ = lean_box(0);
v___x_631_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_631_, 0, v___x_630_);
lean_ctor_set(v___x_631_, 1, v___x_629_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0(lean_object* v_x_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_){
_start:
{
lean_object* v___x_637_; lean_object* v___x_638_; 
v___x_637_ = lean_obj_once(&lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___closed__0, &lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___closed__0_once, _init_lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___closed__0);
v___x_638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_638_, 0, v___x_637_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0___boxed(lean_object* v_x_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___lam__0(v_x_639_, v___y_640_, v___y_641_, v___y_642_);
lean_dec(v___y_642_);
lean_dec_ref(v___y_641_);
lean_dec(v___y_640_);
lean_dec(v_x_639_);
return v_res_644_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default(lean_object* v_00_u03b1_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = ((lean_object*)(lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default___closed__5));
return v___x_659_;
}
}
static lean_object* _init_lp_batteries_Lean_instInhabitedNameMapAttributeImpl___closed__0(void){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lp_batteries_Lean_instInhabitedNameMapAttributeImpl_default(lean_box(0));
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instInhabitedNameMapAttributeImpl(lean_object* v_a_661_){
_start:
{
lean_object* v___x_662_; 
v___x_662_ = lean_obj_once(&lp_batteries_Lean_instInhabitedNameMapAttributeImpl___closed__0, &lp_batteries_Lean_instInhabitedNameMapAttributeImpl___closed__0_once, _init_lp_batteries_Lean_instInhabitedNameMapAttributeImpl___closed__0);
return v___x_662_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_663_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_664_; lean_object* v___x_665_; 
v___x_664_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__0, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__0_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__0);
v___x_665_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_665_, 0, v___x_664_);
return v___x_665_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__2(void){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; 
v___x_666_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1);
v___x_667_ = lean_unsigned_to_nat(0u);
v___x_668_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_668_, 0, v___x_667_);
lean_ctor_set(v___x_668_, 1, v___x_667_);
lean_ctor_set(v___x_668_, 2, v___x_667_);
lean_ctor_set(v___x_668_, 3, v___x_667_);
lean_ctor_set(v___x_668_, 4, v___x_666_);
lean_ctor_set(v___x_668_, 5, v___x_666_);
lean_ctor_set(v___x_668_, 6, v___x_666_);
lean_ctor_set(v___x_668_, 7, v___x_666_);
lean_ctor_set(v___x_668_, 8, v___x_666_);
lean_ctor_set(v___x_668_, 9, v___x_666_);
return v___x_668_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__3(void){
_start:
{
lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; 
v___x_669_ = lean_unsigned_to_nat(32u);
v___x_670_ = lean_mk_empty_array_with_capacity(v___x_669_);
v___x_671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_671_, 0, v___x_670_);
return v___x_671_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__4(void){
_start:
{
size_t v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_672_ = ((size_t)5ULL);
v___x_673_ = lean_unsigned_to_nat(0u);
v___x_674_ = lean_unsigned_to_nat(32u);
v___x_675_ = lean_mk_empty_array_with_capacity(v___x_674_);
v___x_676_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__3, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__3_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__3);
v___x_677_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_677_, 0, v___x_676_);
lean_ctor_set(v___x_677_, 1, v___x_675_);
lean_ctor_set(v___x_677_, 2, v___x_673_);
lean_ctor_set(v___x_677_, 3, v___x_673_);
lean_ctor_set_usize(v___x_677_, 4, v___x_672_);
return v___x_677_;
}
}
static lean_object* _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__5(void){
_start:
{
lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; 
v___x_678_ = lean_box(1);
v___x_679_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__4, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__4_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__4);
v___x_680_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__1);
v___x_681_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_681_, 0, v___x_680_);
lean_ctor_set(v___x_681_, 1, v___x_679_);
lean_ctor_set(v___x_681_, 2, v___x_678_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2(lean_object* v_msgData_682_, lean_object* v___y_683_, lean_object* v___y_684_){
_start:
{
lean_object* v___x_686_; lean_object* v_env_687_; lean_object* v_options_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; 
v___x_686_ = lean_st_ref_get(v___y_684_);
v_env_687_ = lean_ctor_get(v___x_686_, 0);
lean_inc_ref(v_env_687_);
lean_dec(v___x_686_);
v_options_688_ = lean_ctor_get(v___y_683_, 2);
v___x_689_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__2, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__2_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__2);
v___x_690_ = lean_obj_once(&lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__5, &lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__5_once, _init_lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___closed__5);
lean_inc_ref(v_options_688_);
v___x_691_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_691_, 0, v_env_687_);
lean_ctor_set(v___x_691_, 1, v___x_689_);
lean_ctor_set(v___x_691_, 2, v___x_690_);
lean_ctor_set(v___x_691_, 3, v_options_688_);
v___x_692_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_692_, 0, v___x_691_);
lean_ctor_set(v___x_692_, 1, v_msgData_682_);
v___x_693_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_693_, 0, v___x_692_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2___boxed(lean_object* v_msgData_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_){
_start:
{
lean_object* v_res_698_; 
v_res_698_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2(v_msgData_694_, v___y_695_, v___y_696_);
lean_dec(v___y_696_);
lean_dec_ref(v___y_695_);
return v_res_698_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___redArg(lean_object* v_msg_699_, lean_object* v___y_700_, lean_object* v___y_701_){
_start:
{
lean_object* v_ref_703_; lean_object* v___x_704_; lean_object* v_a_705_; lean_object* v___x_707_; uint8_t v_isShared_708_; uint8_t v_isSharedCheck_713_; 
v_ref_703_ = lean_ctor_get(v___y_700_, 5);
v___x_704_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1_spec__2(v_msg_699_, v___y_700_, v___y_701_);
v_a_705_ = lean_ctor_get(v___x_704_, 0);
v_isSharedCheck_713_ = !lean_is_exclusive(v___x_704_);
if (v_isSharedCheck_713_ == 0)
{
v___x_707_ = v___x_704_;
v_isShared_708_ = v_isSharedCheck_713_;
goto v_resetjp_706_;
}
else
{
lean_inc(v_a_705_);
lean_dec(v___x_704_);
v___x_707_ = lean_box(0);
v_isShared_708_ = v_isSharedCheck_713_;
goto v_resetjp_706_;
}
v_resetjp_706_:
{
lean_object* v___x_709_; lean_object* v___x_711_; 
lean_inc(v_ref_703_);
v___x_709_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_709_, 0, v_ref_703_);
lean_ctor_set(v___x_709_, 1, v_a_705_);
if (v_isShared_708_ == 0)
{
lean_ctor_set_tag(v___x_707_, 1);
lean_ctor_set(v___x_707_, 0, v___x_709_);
v___x_711_ = v___x_707_;
goto v_reusejp_710_;
}
else
{
lean_object* v_reuseFailAlloc_712_; 
v_reuseFailAlloc_712_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_712_, 0, v___x_709_);
v___x_711_ = v_reuseFailAlloc_712_;
goto v_reusejp_710_;
}
v_reusejp_710_:
{
return v___x_711_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___redArg___boxed(lean_object* v_msg_714_, lean_object* v___y_715_, lean_object* v___y_716_, lean_object* v___y_717_){
_start:
{
lean_object* v_res_718_; 
v_res_718_ = lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___redArg(v_msg_714_, v___y_715_, v___y_716_);
lean_dec(v___y_716_);
lean_dec_ref(v___y_715_);
return v_res_718_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_720_; lean_object* v___x_721_; 
v___x_720_ = ((lean_object*)(lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__0));
v___x_721_ = l_Lean_stringToMessageData(v___x_720_);
return v___x_721_;
}
}
static lean_object* _init_lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__3(void){
_start:
{
lean_object* v___x_723_; lean_object* v___x_724_; 
v___x_723_ = ((lean_object*)(lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__2));
v___x_724_ = l_Lean_stringToMessageData(v___x_723_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0(lean_object* v_name_725_, lean_object* v_decl_726_, lean_object* v___y_727_, lean_object* v___y_728_){
_start:
{
lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; 
v___x_730_ = lean_obj_once(&lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__1, &lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__1_once, _init_lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__1);
v___x_731_ = l_Lean_MessageData_ofName(v_name_725_);
v___x_732_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_732_, 0, v___x_730_);
lean_ctor_set(v___x_732_, 1, v___x_731_);
v___x_733_ = lean_obj_once(&lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__3, &lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__3_once, _init_lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___closed__3);
v___x_734_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_734_, 0, v___x_732_);
lean_ctor_set(v___x_734_, 1, v___x_733_);
v___x_735_ = lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___redArg(v___x_734_, v___y_727_, v___y_728_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___boxed(lean_object* v_name_736_, lean_object* v_decl_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_){
_start:
{
lean_object* v_res_741_; 
v_res_741_ = lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0(v_name_736_, v_decl_737_, v___y_738_, v___y_739_);
lean_dec(v___y_739_);
lean_dec_ref(v___y_738_);
lean_dec(v_decl_737_);
return v_res_741_;
}
}
static lean_object* _init_lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_742_; 
v___x_742_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_742_;
}
}
static lean_object* _init_lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_743_; lean_object* v___x_744_; 
v___x_743_ = lean_obj_once(&lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__0, &lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__0);
v___x_744_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_744_, 0, v___x_743_);
return v___x_744_;
}
}
static lean_object* _init_lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_745_; lean_object* v___x_746_; 
v___x_745_ = lean_obj_once(&lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__1, &lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__1_once, _init_lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__1);
v___x_746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_746_, 0, v___x_745_);
lean_ctor_set(v___x_746_, 1, v___x_745_);
return v___x_746_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg(lean_object* v_env_747_, lean_object* v___y_748_){
_start:
{
lean_object* v___x_750_; lean_object* v_nextMacroScope_751_; lean_object* v_ngen_752_; lean_object* v_auxDeclNGen_753_; lean_object* v_traceState_754_; lean_object* v_messages_755_; lean_object* v_infoState_756_; lean_object* v_snapshotTasks_757_; lean_object* v___x_759_; uint8_t v_isShared_760_; uint8_t v_isSharedCheck_768_; 
v___x_750_ = lean_st_ref_take(v___y_748_);
v_nextMacroScope_751_ = lean_ctor_get(v___x_750_, 1);
v_ngen_752_ = lean_ctor_get(v___x_750_, 2);
v_auxDeclNGen_753_ = lean_ctor_get(v___x_750_, 3);
v_traceState_754_ = lean_ctor_get(v___x_750_, 4);
v_messages_755_ = lean_ctor_get(v___x_750_, 6);
v_infoState_756_ = lean_ctor_get(v___x_750_, 7);
v_snapshotTasks_757_ = lean_ctor_get(v___x_750_, 8);
v_isSharedCheck_768_ = !lean_is_exclusive(v___x_750_);
if (v_isSharedCheck_768_ == 0)
{
lean_object* v_unused_769_; lean_object* v_unused_770_; 
v_unused_769_ = lean_ctor_get(v___x_750_, 5);
lean_dec(v_unused_769_);
v_unused_770_ = lean_ctor_get(v___x_750_, 0);
lean_dec(v_unused_770_);
v___x_759_ = v___x_750_;
v_isShared_760_ = v_isSharedCheck_768_;
goto v_resetjp_758_;
}
else
{
lean_inc(v_snapshotTasks_757_);
lean_inc(v_infoState_756_);
lean_inc(v_messages_755_);
lean_inc(v_traceState_754_);
lean_inc(v_auxDeclNGen_753_);
lean_inc(v_ngen_752_);
lean_inc(v_nextMacroScope_751_);
lean_dec(v___x_750_);
v___x_759_ = lean_box(0);
v_isShared_760_ = v_isSharedCheck_768_;
goto v_resetjp_758_;
}
v_resetjp_758_:
{
lean_object* v___x_761_; lean_object* v___x_763_; 
v___x_761_ = lean_obj_once(&lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__2, &lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__2_once, _init_lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___closed__2);
if (v_isShared_760_ == 0)
{
lean_ctor_set(v___x_759_, 5, v___x_761_);
lean_ctor_set(v___x_759_, 0, v_env_747_);
v___x_763_ = v___x_759_;
goto v_reusejp_762_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v_env_747_);
lean_ctor_set(v_reuseFailAlloc_767_, 1, v_nextMacroScope_751_);
lean_ctor_set(v_reuseFailAlloc_767_, 2, v_ngen_752_);
lean_ctor_set(v_reuseFailAlloc_767_, 3, v_auxDeclNGen_753_);
lean_ctor_set(v_reuseFailAlloc_767_, 4, v_traceState_754_);
lean_ctor_set(v_reuseFailAlloc_767_, 5, v___x_761_);
lean_ctor_set(v_reuseFailAlloc_767_, 6, v_messages_755_);
lean_ctor_set(v_reuseFailAlloc_767_, 7, v_infoState_756_);
lean_ctor_set(v_reuseFailAlloc_767_, 8, v_snapshotTasks_757_);
v___x_763_ = v_reuseFailAlloc_767_;
goto v_reusejp_762_;
}
v_reusejp_762_:
{
lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; 
v___x_764_ = lean_st_ref_set(v___y_748_, v___x_763_);
v___x_765_ = lean_box(0);
v___x_766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_766_, 0, v___x_765_);
return v___x_766_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg___boxed(lean_object* v_env_771_, lean_object* v___y_772_, lean_object* v___y_773_){
_start:
{
lean_object* v_res_774_; 
v_res_774_ = lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg(v_env_771_, v___y_772_);
lean_dec(v___y_772_);
return v_res_774_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___redArg(lean_object* v_ext_775_, lean_object* v_k_776_, lean_object* v_v_777_, lean_object* v___y_778_, lean_object* v___y_779_){
_start:
{
lean_object* v___x_781_; lean_object* v_env_782_; lean_object* v___x_783_; 
v___x_781_ = lean_st_ref_get(v___y_779_);
v_env_782_ = lean_ctor_get(v___x_781_, 0);
lean_inc_ref(v_env_782_);
lean_dec(v___x_781_);
v___x_783_ = lp_batteries_Lean_NameMapExtension_find_x3f___redArg(v_ext_775_, v_env_782_, v_k_776_);
if (lean_obj_tag(v___x_783_) == 1)
{
lean_object* v_name_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; 
lean_dec_ref_known(v___x_783_, 1);
lean_dec(v_v_777_);
v_name_784_ = lean_ctor_get(v_ext_775_, 1);
lean_inc(v_name_784_);
lean_dec_ref(v_ext_775_);
v___x_785_ = lean_obj_once(&lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1, &lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1_once, _init_lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__1);
v___x_786_ = l_Lean_MessageData_ofName(v_name_784_);
v___x_787_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_787_, 0, v___x_785_);
lean_ctor_set(v___x_787_, 1, v___x_786_);
v___x_788_ = lean_obj_once(&lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3, &lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3_once, _init_lp_batteries_Lean_NameMapExtension_add___redArg___lam__1___closed__3);
v___x_789_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_789_, 0, v___x_787_);
lean_ctor_set(v___x_789_, 1, v___x_788_);
v___x_790_ = l_Lean_MessageData_ofName(v_k_776_);
v___x_791_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_791_, 0, v___x_789_);
lean_ctor_set(v___x_791_, 1, v___x_790_);
v___x_792_ = lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___redArg(v___x_791_, v___y_778_, v___y_779_);
return v___x_792_;
}
else
{
lean_object* v___x_793_; lean_object* v_toEnvExtension_794_; lean_object* v_env_795_; lean_object* v_asyncMode_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; 
lean_dec(v___x_783_);
v___x_793_ = lean_st_ref_get(v___y_779_);
v_toEnvExtension_794_ = lean_ctor_get(v_ext_775_, 0);
v_env_795_ = lean_ctor_get(v___x_793_, 0);
lean_inc_ref(v_env_795_);
lean_dec(v___x_793_);
v_asyncMode_796_ = lean_ctor_get(v_toEnvExtension_794_, 2);
lean_inc(v_asyncMode_796_);
v___x_797_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_797_, 0, v_k_776_);
lean_ctor_set(v___x_797_, 1, v_v_777_);
v___x_798_ = lean_box(0);
v___x_799_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v_ext_775_, v_env_795_, v___x_797_, v_asyncMode_796_, v___x_798_);
lean_dec(v_asyncMode_796_);
v___x_800_ = lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg(v___x_799_, v___y_779_);
return v___x_800_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___redArg___boxed(lean_object* v_ext_801_, lean_object* v_k_802_, lean_object* v_v_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_){
_start:
{
lean_object* v_res_807_; 
v_res_807_ = lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___redArg(v_ext_801_, v_k_802_, v_v_803_, v___y_804_, v___y_805_);
lean_dec(v___y_805_);
lean_dec_ref(v___y_804_);
return v_res_807_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__1(lean_object* v_add_808_, lean_object* v_a_809_, lean_object* v_src_810_, lean_object* v_stx_811_, uint8_t v___kind_812_, lean_object* v___y_813_, lean_object* v___y_814_){
_start:
{
lean_object* v___x_816_; 
lean_inc(v___y_814_);
lean_inc_ref(v___y_813_);
lean_inc(v_src_810_);
v___x_816_ = lean_apply_5(v_add_808_, v_src_810_, v_stx_811_, v___y_813_, v___y_814_, lean_box(0));
if (lean_obj_tag(v___x_816_) == 0)
{
lean_object* v_a_817_; lean_object* v___x_818_; 
v_a_817_ = lean_ctor_get(v___x_816_, 0);
lean_inc(v_a_817_);
lean_dec_ref_known(v___x_816_, 1);
v___x_818_ = lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___redArg(v_a_809_, v_src_810_, v_a_817_, v___y_813_, v___y_814_);
return v___x_818_;
}
else
{
lean_object* v_a_819_; lean_object* v___x_821_; uint8_t v_isShared_822_; uint8_t v_isSharedCheck_826_; 
lean_dec(v_src_810_);
lean_dec_ref(v_a_809_);
v_a_819_ = lean_ctor_get(v___x_816_, 0);
v_isSharedCheck_826_ = !lean_is_exclusive(v___x_816_);
if (v_isSharedCheck_826_ == 0)
{
v___x_821_ = v___x_816_;
v_isShared_822_ = v_isSharedCheck_826_;
goto v_resetjp_820_;
}
else
{
lean_inc(v_a_819_);
lean_dec(v___x_816_);
v___x_821_ = lean_box(0);
v_isShared_822_ = v_isSharedCheck_826_;
goto v_resetjp_820_;
}
v_resetjp_820_:
{
lean_object* v___x_824_; 
if (v_isShared_822_ == 0)
{
v___x_824_ = v___x_821_;
goto v_reusejp_823_;
}
else
{
lean_object* v_reuseFailAlloc_825_; 
v_reuseFailAlloc_825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_825_, 0, v_a_819_);
v___x_824_ = v_reuseFailAlloc_825_;
goto v_reusejp_823_;
}
v_reusejp_823_:
{
return v___x_824_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___lam__1___boxed(lean_object* v_add_827_, lean_object* v_a_828_, lean_object* v_src_829_, lean_object* v_stx_830_, lean_object* v___kind_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_){
_start:
{
uint8_t v___kind_boxed_835_; lean_object* v_res_836_; 
v___kind_boxed_835_ = lean_unbox(v___kind_831_);
v_res_836_ = lp_batteries_Lean_registerNameMapAttribute___redArg___lam__1(v_add_827_, v_a_828_, v_src_829_, v_stx_830_, v___kind_boxed_835_, v___y_832_, v___y_833_);
lean_dec(v___y_833_);
lean_dec_ref(v___y_832_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg(lean_object* v_impl_841_){
_start:
{
lean_object* v_name_843_; lean_object* v_ref_844_; lean_object* v_descr_845_; lean_object* v_add_846_; lean_object* v___x_847_; 
v_name_843_ = lean_ctor_get(v_impl_841_, 0);
lean_inc(v_name_843_);
v_ref_844_ = lean_ctor_get(v_impl_841_, 1);
lean_inc(v_ref_844_);
v_descr_845_ = lean_ctor_get(v_impl_841_, 2);
lean_inc_ref(v_descr_845_);
v_add_846_ = lean_ctor_get(v_impl_841_, 3);
lean_inc_ref(v_add_846_);
lean_dec_ref(v_impl_841_);
v___x_847_ = lp_batteries_Lean_registerNameMapExtension___redArg(v_ref_844_);
if (lean_obj_tag(v___x_847_) == 0)
{
lean_object* v_a_848_; lean_object* v___f_849_; lean_object* v___f_850_; lean_object* v___x_851_; uint8_t v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; 
v_a_848_ = lean_ctor_get(v___x_847_, 0);
lean_inc_n(v_a_848_, 2);
lean_dec_ref_known(v___x_847_, 1);
lean_inc(v_name_843_);
v___f_849_ = lean_alloc_closure((void*)(lp_batteries_Lean_registerNameMapAttribute___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_849_, 0, v_name_843_);
v___f_850_ = lean_alloc_closure((void*)(lp_batteries_Lean_registerNameMapAttribute___redArg___lam__1___boxed), 8, 2);
lean_closure_set(v___f_850_, 0, v_add_846_);
lean_closure_set(v___f_850_, 1, v_a_848_);
v___x_851_ = ((lean_object*)(lp_batteries_Lean_registerNameMapAttribute___redArg___closed__1));
v___x_852_ = 0;
v___x_853_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_853_, 0, v___x_851_);
lean_ctor_set(v___x_853_, 1, v_name_843_);
lean_ctor_set(v___x_853_, 2, v_descr_845_);
lean_ctor_set_uint8(v___x_853_, sizeof(void*)*3, v___x_852_);
v___x_854_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_854_, 0, v___x_853_);
lean_ctor_set(v___x_854_, 1, v___f_850_);
lean_ctor_set(v___x_854_, 2, v___f_849_);
v___x_855_ = l_Lean_registerBuiltinAttribute(v___x_854_);
if (lean_obj_tag(v___x_855_) == 0)
{
lean_object* v___x_857_; uint8_t v_isShared_858_; uint8_t v_isSharedCheck_862_; 
v_isSharedCheck_862_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_862_ == 0)
{
lean_object* v_unused_863_; 
v_unused_863_ = lean_ctor_get(v___x_855_, 0);
lean_dec(v_unused_863_);
v___x_857_ = v___x_855_;
v_isShared_858_ = v_isSharedCheck_862_;
goto v_resetjp_856_;
}
else
{
lean_dec(v___x_855_);
v___x_857_ = lean_box(0);
v_isShared_858_ = v_isSharedCheck_862_;
goto v_resetjp_856_;
}
v_resetjp_856_:
{
lean_object* v___x_860_; 
if (v_isShared_858_ == 0)
{
lean_ctor_set(v___x_857_, 0, v_a_848_);
v___x_860_ = v___x_857_;
goto v_reusejp_859_;
}
else
{
lean_object* v_reuseFailAlloc_861_; 
v_reuseFailAlloc_861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_861_, 0, v_a_848_);
v___x_860_ = v_reuseFailAlloc_861_;
goto v_reusejp_859_;
}
v_reusejp_859_:
{
return v___x_860_;
}
}
}
else
{
lean_object* v_a_864_; lean_object* v___x_866_; uint8_t v_isShared_867_; uint8_t v_isSharedCheck_871_; 
lean_dec(v_a_848_);
v_a_864_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_871_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_871_ == 0)
{
v___x_866_ = v___x_855_;
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
else
{
lean_inc(v_a_864_);
lean_dec(v___x_855_);
v___x_866_ = lean_box(0);
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
v_resetjp_865_:
{
lean_object* v___x_869_; 
if (v_isShared_867_ == 0)
{
v___x_869_ = v___x_866_;
goto v_reusejp_868_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v_a_864_);
v___x_869_ = v_reuseFailAlloc_870_;
goto v_reusejp_868_;
}
v_reusejp_868_:
{
return v___x_869_;
}
}
}
}
else
{
lean_dec_ref(v_add_846_);
lean_dec_ref(v_descr_845_);
lean_dec(v_name_843_);
return v___x_847_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___redArg___boxed(lean_object* v_impl_872_, lean_object* v_a_873_){
_start:
{
lean_object* v_res_874_; 
v_res_874_ = lp_batteries_Lean_registerNameMapAttribute___redArg(v_impl_872_);
return v_res_874_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute(lean_object* v_00_u03b1_875_, lean_object* v_impl_876_){
_start:
{
lean_object* v___x_878_; 
v___x_878_ = lp_batteries_Lean_registerNameMapAttribute___redArg(v_impl_876_);
return v___x_878_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_registerNameMapAttribute___boxed(lean_object* v_00_u03b1_879_, lean_object* v_impl_880_, lean_object* v_a_881_){
_start:
{
lean_object* v_res_882_; 
v_res_882_ = lp_batteries_Lean_registerNameMapAttribute(v_00_u03b1_879_, v_impl_880_);
return v_res_882_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0(lean_object* v_env_883_, lean_object* v___y_884_, lean_object* v___y_885_){
_start:
{
lean_object* v___x_887_; 
v___x_887_ = lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___redArg(v_env_883_, v___y_885_);
return v___x_887_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0___boxed(lean_object* v_env_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_){
_start:
{
lean_object* v_res_892_; 
v_res_892_ = lp_batteries_Lean_setEnv___at___00Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0_spec__0(v_env_888_, v___y_889_, v___y_890_);
lean_dec(v___y_890_);
lean_dec_ref(v___y_889_);
return v_res_892_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0(lean_object* v_00_u03b1_893_, lean_object* v_ext_894_, lean_object* v_k_895_, lean_object* v_v_896_, lean_object* v___y_897_, lean_object* v___y_898_){
_start:
{
lean_object* v___x_900_; 
v___x_900_ = lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___redArg(v_ext_894_, v_k_895_, v_v_896_, v___y_897_, v___y_898_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0___boxed(lean_object* v_00_u03b1_901_, lean_object* v_ext_902_, lean_object* v_k_903_, lean_object* v_v_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_){
_start:
{
lean_object* v_res_908_; 
v_res_908_ = lp_batteries_Lean_NameMapExtension_add___at___00Lean_registerNameMapAttribute_spec__0(v_00_u03b1_901_, v_ext_902_, v_k_903_, v_v_904_, v___y_905_, v___y_906_);
lean_dec(v___y_906_);
lean_dec_ref(v___y_905_);
return v_res_908_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1(lean_object* v_00_u03b1_909_, lean_object* v_msg_910_, lean_object* v___y_911_, lean_object* v___y_912_){
_start:
{
lean_object* v___x_914_; 
v___x_914_ = lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___redArg(v_msg_910_, v___y_911_, v___y_912_);
return v___x_914_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1___boxed(lean_object* v_00_u03b1_915_, lean_object* v_msg_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_){
_start:
{
lean_object* v_res_920_; 
v_res_920_ = lp_batteries_Lean_throwError___at___00Lean_registerNameMapAttribute_spec__1(v_00_u03b1_915_, v_msg_916_, v___y_917_, v___y_918_);
lean_dec(v___y_918_);
lean_dec_ref(v___y_917_);
return v_res_920_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Attributes(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_NameMapAttribute(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_NameMapAttribute(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Lean_registerNameMapExtension___auto__1 = _init_lp_batteries_Lean_registerNameMapExtension___auto__1();
lean_mark_persistent(lp_batteries_Lean_registerNameMapExtension___auto__1);
lp_batteries_Lean_NameMapAttributeImpl_ref___autoParam = _init_lp_batteries_Lean_NameMapAttributeImpl_ref___autoParam();
lean_mark_persistent(lp_batteries_Lean_NameMapAttributeImpl_ref___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Attributes(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_NameMapAttribute(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_NameMapAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_NameMapAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_NameMapAttribute(builtin);
}
#ifdef __cplusplus
}
#endif
