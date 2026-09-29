// Lean compiler output
// Module: Batteries.Data.List.Basic
// Imports: public import Init public meta import Init public import Batteries.Tactic.Alias
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
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_List_drop___redArg(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l___private_Init_Data_List_Impl_0__List_setTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_filterTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_List_insert(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_flip(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_splitAt___redArg(lean_object*, lean_object*);
uint8_t l_List_beq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* l_List_forM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_eraseTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverseAux___redArg(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_List_countP_go___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_List_decidableBAll___redArg(lean_object*, lean_object*);
lean_object* l_List_zipWith___at___00List_zip_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_push___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instFunctorOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Option_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Option_bind(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_concat___redArg(lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_max_x3f___redArg(lean_object*, lean_object*);
lean_object* l_panic___redArg(lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__0(lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l_List_mapM_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_min_x3f___redArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_max_x21___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Batteries.Data.List.Basic"};
static const lean_object* lp_batteries_List_max_x21___redArg___closed__0 = (const lean_object*)&lp_batteries_List_max_x21___redArg___closed__0_value;
static const lean_string_object lp_batteries_List_max_x21___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "List.max!"};
static const lean_object* lp_batteries_List_max_x21___redArg___closed__1 = (const lean_object*)&lp_batteries_List_max_x21___redArg___closed__1_value;
static const lean_string_object lp_batteries_List_max_x21___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "List.max! called on empty list"};
static const lean_object* lp_batteries_List_max_x21___redArg___closed__2 = (const lean_object*)&lp_batteries_List_max_x21___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_List_max_x21___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_max_x21___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries_List_max_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_max_x21___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_max_x21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_max_x21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_min_x21___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "List.min!"};
static const lean_object* lp_batteries_List_min_x21___redArg___closed__0 = (const lean_object*)&lp_batteries_List_min_x21___redArg___closed__0_value;
static const lean_string_object lp_batteries_List_min_x21___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "List.min! called on empty list"};
static const lean_object* lp_batteries_List_min_x21___redArg___closed__1 = (const lean_object*)&lp_batteries_List_min_x21___redArg___closed__1_value;
static lean_once_cell_t lp_batteries_List_min_x21___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List_min_x21___redArg___closed__2;
LEAN_EXPORT lean_object* lp_batteries_List_min_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_min_x21___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_min_x21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_min_x21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_List_bagInter___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_List_bagInter___redArg___closed__0 = (const lean_object*)&lp_batteries_List_bagInter___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_bagInter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_bagInter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_diff___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_diff(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_next_x3f___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_next_x3f(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_after___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_after(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_replaceF___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_replaceF(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_next_x3f_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_next_x3f_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_max_x21_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_max_x21_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR_go___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_List_replaceFTR_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__0 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__0_value;
static const lean_closure_object lp_batteries_List_replaceFTR_go___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__1 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__1_value;
static const lean_closure_object lp_batteries_List_replaceFTR_go___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__2 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__2_value;
static const lean_closure_object lp_batteries_List_replaceFTR_go___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__3 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__3_value;
static const lean_closure_object lp_batteries_List_replaceFTR_go___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__4 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__4_value;
static const lean_closure_object lp_batteries_List_replaceFTR_go___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__5 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__5_value;
static const lean_closure_object lp_batteries_List_replaceFTR_go___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__6 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__6_value;
static const lean_ctor_object lp_batteries_List_replaceFTR_go___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__0_value),((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__1_value)}};
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__7 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__7_value;
static const lean_ctor_object lp_batteries_List_replaceFTR_go___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__7_value),((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__2_value),((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__3_value),((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__4_value),((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__5_value)}};
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__8 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__8_value;
static const lean_ctor_object lp_batteries_List_replaceFTR_go___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__8_value),((lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__6_value)}};
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__9 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__9_value;
static const lean_closure_object lp_batteries_List_replaceFTR_go___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_List_replaceFTR_go___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_replaceFTR_go___redArg___closed__10 = (const lean_object*)&lp_batteries_List_replaceFTR_go___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_replaceFTR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_replaceFTR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_union___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_union(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instUnionOfBEq__batteries___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instUnionOfBEq__batteries(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_inter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_inter___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_inter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_inter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instInterOfBEq__batteries___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instInterOfBEq__batteries(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_splitAtD_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_splitAtD_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_splitAtD___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_splitAtD(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_modifyLast_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_modifyLast_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_modifyLast___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_modifyLast(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeD___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeD___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeD(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeD___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeDTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeDTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeDTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeDTR(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeDTR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeDTR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldlIdx___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldlIdx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdx___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdx___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_foldlIdx_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_foldlIdx_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxNth_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxNth_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxNth___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_findIdxNth(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_idxOfNth___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxOfNth___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxOfNth___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_idxOfNth(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore_go___at___00List_countPBefore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore_go___at___00List_countPBefore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_countBefore___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_countBefore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_lookmap_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_lookmap_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_lookmap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_lookmap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_inits_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_inits___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_inits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_inits_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_initsTR_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_List_initsTR___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_List_initsTR___redArg___closed__0 = (const lean_object*)&lp_batteries_List_initsTR___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_initsTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_initsTR(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_initsTR_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_tails___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_tails(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_tailsTR_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_tailsTR_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_List_tailsTR___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_List_tailsTR___redArg___closed__0 = (const lean_object*)&lp_batteries_List_tailsTR___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_tailsTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_tailsTR(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublists_x27_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sublists_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sublists_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublists_x27_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sublists_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublists_spec__1___redArg(lean_object*, lean_object*);
static const lean_ctor_object lp_batteries_List_sublists___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_List_sublists___redArg___closed__0 = (const lean_object*)&lp_batteries_List_sublists___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_sublists___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sublists(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sublists_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublists_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublistsFast_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sublistsFast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sublistsFast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublistsFast_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_all_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_all_u2082___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_all_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_all_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_all_u2082_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_all_u2082_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableForall_u2082___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableForall_u2082___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableForall_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableForall_u2082___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableForall_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableForall_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_transpose_pop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_transpose_pop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___redArg(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_transpose_go_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_transpose_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_transpose_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_transpose_go_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_transpose_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_transpose___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_transpose(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_transpose_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_sections_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sections_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sections___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sections(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_sections_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sections_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_sections_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_sections_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sectionsTR_go_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR_go___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR_go___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sectionsTR_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00List_sectionsTR_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00List_sectionsTR_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sectionsTR_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00List_sectionsTR_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00List_sectionsTR_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sectionsTR_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_any_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_any_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_extractP_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_extractP_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_extractP___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_extractP(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_revzip___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_revzip(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_product_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_product_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_List_product___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_List_product___redArg___closed__0 = (const lean_object*)&lp_batteries_List_product___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_product___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_product(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_product_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_product_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_productTR_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_productTR_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_productTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_productTR(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_productTR_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_productTR_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_sigma_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sigma_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_List_sigma___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_List_sigma___redArg___closed__0 = (const lean_object*)&lp_batteries_List_sigma___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_sigma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sigma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_sigma_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sigma_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sigmaTR_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sigmaTR_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sigmaTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_sigmaTR(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sigmaTR_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sigmaTR_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_ofFnNthVal___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_ofFnNthVal___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_ofFnNthVal(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_ofFnNthVal___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries_List_takeWhile_u2082___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_List_takeWhile_u2082___redArg___closed__0 = (const lean_object*)&lp_batteries_List_takeWhile_u2082___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082TR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082TR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082TR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082TR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082TR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082TR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_pwFilter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_pwFilter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_pwFilter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableIsChainOfDecidableRel_go___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableIsChainOfDecidableRel_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableIsChainOfDecidableRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_eraseDup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_eraseDup___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_eraseDup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_eraseDup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_rotate___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_rotate___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_rotate(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_rotate___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_rotate_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_rotate_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_rotate_x27_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_rotate_x27_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forDiagM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forDiagM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forDiagM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_forDiagM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_getRest___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_getRest(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSlice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSlice___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSlice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSlice___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSlice_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSlice_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_zipWithLeft_x27_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_zipWithLeft_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_zipWithLeft_x27_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_zipWithLeft_x27_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_zipWithLeft_x27TR_go_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27TR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27TR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_zipWithLeft_x27TR_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27TR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27TR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_zipWithLeft_x27TR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_zipWithLeft_x27TR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft_x27___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_List_zipLeft_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_List_zipLeft_x27___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_zipLeft_x27___redArg___closed__0 = (const lean_object*)&lp_batteries_List_zipLeft_x27___redArg___closed__0_value;
static const lean_array_object lp_batteries_List_zipLeft_x27___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_List_zipLeft_x27___redArg___closed__1 = (const lean_object*)&lp_batteries_List_zipLeft_x27___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipRight_x27___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_List_zipRight_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_List_zipRight_x27___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_zipRight_x27___redArg___closed__0 = (const lean_object*)&lp_batteries_List_zipRight_x27___redArg___closed__0_value;
static const lean_closure_object lp_batteries_List_zipRight_x27___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_flip, .m_arity = 6, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List_zipRight_x27___redArg___closed__0_value)} };
static const lean_object* lp_batteries_List_zipRight_x27___redArg___closed__1 = (const lean_object*)&lp_batteries_List_zipRight_x27___redArg___closed__1_value;
static const lean_array_object lp_batteries_List_zipRight_x27___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_List_zipRight_x27___redArg___closed__2 = (const lean_object*)&lp_batteries_List_zipRight_x27___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_List_zipRight_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipRight_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_List_allSome___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_allSome___redArg___closed__0 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__0_value;
static const lean_closure_object lp_batteries_List_allSome___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_allSome___redArg___closed__1 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__1_value;
static const lean_closure_object lp_batteries_List_allSome___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_allSome___redArg___closed__2 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__2_value;
static const lean_closure_object lp_batteries_List_allSome___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__3___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_allSome___redArg___closed__3 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__3_value;
static const lean_closure_object lp_batteries_List_allSome___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instFunctorOption___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_allSome___redArg___closed__4 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__4_value;
static const lean_closure_object lp_batteries_List_allSome___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_map, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_allSome___redArg___closed__5 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__5_value;
static const lean_ctor_object lp_batteries_List_allSome___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_List_allSome___redArg___closed__5_value),((lean_object*)&lp_batteries_List_allSome___redArg___closed__4_value)}};
static const lean_object* lp_batteries_List_allSome___redArg___closed__6 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__6_value;
static const lean_ctor_object lp_batteries_List_allSome___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_List_allSome___redArg___closed__6_value),((lean_object*)&lp_batteries_List_allSome___redArg___closed__0_value),((lean_object*)&lp_batteries_List_allSome___redArg___closed__1_value),((lean_object*)&lp_batteries_List_allSome___redArg___closed__2_value),((lean_object*)&lp_batteries_List_allSome___redArg___closed__3_value)}};
static const lean_object* lp_batteries_List_allSome___redArg___closed__7 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__7_value;
static const lean_closure_object lp_batteries_List_allSome___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_bind, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_allSome___redArg___closed__8 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__8_value;
static const lean_ctor_object lp_batteries_List_allSome___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_List_allSome___redArg___closed__7_value),((lean_object*)&lp_batteries_List_allSome___redArg___closed__8_value)}};
static const lean_object* lp_batteries_List_allSome___redArg___closed__9 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__9_value;
static const lean_closure_object lp_batteries_List_allSome___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries_List_allSome___redArg___closed__10 = (const lean_object*)&lp_batteries_List_allSome___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_batteries_List_allSome___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_allSome(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeListTR_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeListTR_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeListTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_takeListTR(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeListTR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeListTR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_rotate_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_rotate_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeList_match__3_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeList_match__3_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeList_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeList_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunksAux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunksAux___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunksAux(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunksAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunks_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunks_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunks_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunks_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunks___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunks___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunks(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_toChunks___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2083___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2083(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2084___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2084(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2085___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2085(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapWithPrefixSuffixAux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapWithPrefixSuffixAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapWithPrefixSuffix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapWithPrefixSuffix(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapWithComplement___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapWithComplement___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapWithComplement(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_traverse___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_List_traverse___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_List_traverse___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_List_traverse___redArg___closed__0 = (const lean_object*)&lp_batteries_List_traverse___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_List_traverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_traverse___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_traverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__0 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__0_value;
static const lean_string_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term_<+~_"};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__1 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__1_value;
static const lean_ctor_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__2_value_aux_0),((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(233, 161, 6, 96, 82, 205, 163, 137)}};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__2 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__2_value;
static const lean_string_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__3 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__3_value;
static const lean_ctor_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__4 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__4_value;
static const lean_string_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " <+~ "};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__5 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__5_value;
static const lean_ctor_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__5_value)}};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__6 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__6_value;
static const lean_string_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__7 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__7_value;
static const lean_ctor_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__8 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__8_value;
static const lean_ctor_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__8_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__9 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__9_value;
static const lean_ctor_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__4_value),((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__6_value),((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__9_value)}};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__10 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__10_value;
static const lean_ctor_object lp_batteries_List_term___x3c_x2b_x7e___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__2_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__10_value)}};
static const lean_object* lp_batteries_List_term___x3c_x2b_x7e___00__closed__11 = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_batteries_List_term___x3c_x2b_x7e__ = (const lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__11_value;
static const lean_string_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__0 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__0_value;
static const lean_string_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__1 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__1_value;
static const lean_string_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__2 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__2_value;
static const lean_string_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__3 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__3_value;
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4_value_aux_0),((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4_value_aux_1),((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4_value_aux_2),((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4_value;
static const lean_string_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Subperm"};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__5 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__5_value;
static lean_once_cell_t lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__6;
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 120, 216, 97, 74, 12, 202, 126)}};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__7 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__7_value;
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List_term___x3c_x2b_x7e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__8_value_aux_0),((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(102, 245, 23, 70, 93, 101, 125, 155)}};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__8 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__8_value;
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__9 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__9_value;
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__10 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__10_value;
static const lean_string_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__11 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__11_value;
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__12 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__12_value;
LEAN_EXPORT lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___closed__0 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___closed__0_value;
static const lean_ctor_object lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___closed__1 = (const lean_object*)&lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_isSubperm___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_isSubperm___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_isSubperm___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_isSubperm___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_List_isSubperm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_isSubperm___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_insertP_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_insertP_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_insertP___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_insertP(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropPrefix_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropPrefix_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSuffix_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropSuffix_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropInfix_x3f_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropInfix_x3f_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropInfix_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_dropInfix_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_partialSums___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_partialSums___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_partialSums(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_partialProds___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_partialProds(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapAt___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapAt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapAtTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapAtTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapAtTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapAtTR(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swap___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swap___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapTR_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapTR_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapTR_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapTR___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_swapTR(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_swapTR_go_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_swapTR_go_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_swap_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_swap_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_List_max_x21___redArg___closed__3(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_4_ = ((lean_object*)(lp_batteries_List_max_x21___redArg___closed__2));
v___x_5_ = lean_unsigned_to_nat(12u);
v___x_6_ = lean_unsigned_to_nat(20u);
v___x_7_ = ((lean_object*)(lp_batteries_List_max_x21___redArg___closed__1));
v___x_8_ = ((lean_object*)(lp_batteries_List_max_x21___redArg___closed__0));
v___x_9_ = l_mkPanicMessageWithDecl(v___x_8_, v___x_7_, v___x_6_, v___x_5_, v___x_4_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_max_x21___redArg(lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_xs_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = l_List_max_x3f___redArg(v_inst_11_, v_xs_12_);
if (lean_obj_tag(v___x_13_) == 0)
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_obj_once(&lp_batteries_List_max_x21___redArg___closed__3, &lp_batteries_List_max_x21___redArg___closed__3_once, _init_lp_batteries_List_max_x21___redArg___closed__3);
v___x_15_ = l_panic___redArg(v_inst_10_, v___x_14_);
return v___x_15_;
}
else
{
lean_object* v_val_16_; 
v_val_16_ = lean_ctor_get(v___x_13_, 0);
lean_inc(v_val_16_);
lean_dec_ref_known(v___x_13_, 1);
return v_val_16_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_max_x21___redArg___boxed(lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_xs_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_batteries_List_max_x21___redArg(v_inst_17_, v_inst_18_, v_xs_19_);
lean_dec(v_inst_17_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_max_x21(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_xs_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lp_batteries_List_max_x21___redArg(v_inst_22_, v_inst_23_, v_xs_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_max_x21___boxed(lean_object* v_00_u03b1_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_xs_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_batteries_List_max_x21(v_00_u03b1_26_, v_inst_27_, v_inst_28_, v_xs_29_);
lean_dec(v_inst_27_);
return v_res_30_;
}
}
static lean_object* _init_lp_batteries_List_min_x21___redArg___closed__2(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_33_ = ((lean_object*)(lp_batteries_List_min_x21___redArg___closed__1));
v___x_34_ = lean_unsigned_to_nat(12u);
v___x_35_ = lean_unsigned_to_nat(27u);
v___x_36_ = ((lean_object*)(lp_batteries_List_min_x21___redArg___closed__0));
v___x_37_ = ((lean_object*)(lp_batteries_List_max_x21___redArg___closed__0));
v___x_38_ = l_mkPanicMessageWithDecl(v___x_37_, v___x_36_, v___x_35_, v___x_34_, v___x_33_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_min_x21___redArg(lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_xs_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = l_List_min_x3f___redArg(v_inst_40_, v_xs_41_);
if (lean_obj_tag(v___x_42_) == 0)
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = lean_obj_once(&lp_batteries_List_min_x21___redArg___closed__2, &lp_batteries_List_min_x21___redArg___closed__2_once, _init_lp_batteries_List_min_x21___redArg___closed__2);
v___x_44_ = l_panic___redArg(v_inst_39_, v___x_43_);
return v___x_44_;
}
else
{
lean_object* v_val_45_; 
v_val_45_ = lean_ctor_get(v___x_42_, 0);
lean_inc(v_val_45_);
lean_dec_ref_known(v___x_42_, 1);
return v_val_45_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_min_x21___redArg___boxed(lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_xs_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_batteries_List_min_x21___redArg(v_inst_46_, v_inst_47_, v_xs_48_);
lean_dec(v_inst_46_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_min_x21(lean_object* v_00_u03b1_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_xs_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_batteries_List_min_x21___redArg(v_inst_51_, v_inst_52_, v_xs_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_min_x21___boxed(lean_object* v_00_u03b1_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_xs_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_batteries_List_min_x21(v_00_u03b1_55_, v_inst_56_, v_inst_57_, v_xs_58_);
lean_dec(v_inst_56_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_bagInter___redArg(lean_object* v_inst_62_, lean_object* v_x_63_, lean_object* v_x_64_){
_start:
{
if (lean_obj_tag(v_x_63_) == 0)
{
lean_dec(v_x_64_);
lean_dec_ref(v_inst_62_);
return v_x_63_;
}
else
{
if (lean_obj_tag(v_x_64_) == 0)
{
lean_dec_ref_known(v_x_63_, 2);
lean_dec_ref(v_inst_62_);
return v_x_64_;
}
else
{
lean_object* v_head_65_; lean_object* v_tail_66_; lean_object* v___x_68_; uint8_t v_isShared_69_; uint8_t v_isSharedCheck_78_; 
v_head_65_ = lean_ctor_get(v_x_63_, 0);
v_tail_66_ = lean_ctor_get(v_x_63_, 1);
v_isSharedCheck_78_ = !lean_is_exclusive(v_x_63_);
if (v_isSharedCheck_78_ == 0)
{
v___x_68_ = v_x_63_;
v_isShared_69_ = v_isSharedCheck_78_;
goto v_resetjp_67_;
}
else
{
lean_inc(v_tail_66_);
lean_inc(v_head_65_);
lean_dec(v_x_63_);
v___x_68_ = lean_box(0);
v_isShared_69_ = v_isSharedCheck_78_;
goto v_resetjp_67_;
}
v_resetjp_67_:
{
uint8_t v___x_70_; 
lean_inc(v_x_64_);
lean_inc(v_head_65_);
lean_inc_ref(v_inst_62_);
v___x_70_ = l_List_elem___redArg(v_inst_62_, v_head_65_, v_x_64_);
if (v___x_70_ == 0)
{
lean_del_object(v___x_68_);
lean_dec(v_head_65_);
v_x_63_ = v_tail_66_;
goto _start;
}
else
{
lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_76_; 
v___x_72_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_head_65_);
lean_inc(v_x_64_);
lean_inc_ref(v_inst_62_);
v___x_73_ = l___private_Init_Data_List_Impl_0__List_eraseTR_go(lean_box(0), v_inst_62_, v_x_64_, v_head_65_, v_x_64_, v___x_72_);
lean_dec(v_x_64_);
v___x_74_ = lp_batteries_List_bagInter___redArg(v_inst_62_, v_tail_66_, v___x_73_);
if (v_isShared_69_ == 0)
{
lean_ctor_set(v___x_68_, 1, v___x_74_);
v___x_76_ = v___x_68_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v_head_65_);
lean_ctor_set(v_reuseFailAlloc_77_, 1, v___x_74_);
v___x_76_ = v_reuseFailAlloc_77_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
return v___x_76_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_bagInter(lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_x_81_, lean_object* v_x_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_batteries_List_bagInter___redArg(v_inst_80_, v_x_81_, v_x_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_diff___redArg(lean_object* v_inst_84_, lean_object* v_x_85_, lean_object* v_x_86_){
_start:
{
if (lean_obj_tag(v_x_86_) == 0)
{
lean_dec_ref(v_inst_84_);
return v_x_85_;
}
else
{
lean_object* v_head_87_; lean_object* v_tail_88_; uint8_t v___x_89_; 
v_head_87_ = lean_ctor_get(v_x_86_, 0);
lean_inc_n(v_head_87_, 2);
v_tail_88_ = lean_ctor_get(v_x_86_, 1);
lean_inc(v_tail_88_);
lean_dec_ref_known(v_x_86_, 2);
lean_inc(v_x_85_);
lean_inc_ref(v_inst_84_);
v___x_89_ = l_List_elem___redArg(v_inst_84_, v_head_87_, v_x_85_);
if (v___x_89_ == 0)
{
lean_dec(v_head_87_);
v_x_86_ = v_tail_88_;
goto _start;
}
else
{
lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_91_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_x_85_);
lean_inc_ref(v_inst_84_);
v___x_92_ = l___private_Init_Data_List_Impl_0__List_eraseTR_go(lean_box(0), v_inst_84_, v_x_85_, v_head_87_, v_x_85_, v___x_91_);
lean_dec(v_x_85_);
v_x_85_ = v___x_92_;
v_x_86_ = v_tail_88_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_diff(lean_object* v_00_u03b1_94_, lean_object* v_inst_95_, lean_object* v_x_96_, lean_object* v_x_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_batteries_List_diff___redArg(v_inst_95_, v_x_96_, v_x_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_next_x3f___redArg(lean_object* v_x_99_){
_start:
{
if (lean_obj_tag(v_x_99_) == 0)
{
lean_object* v___x_100_; 
v___x_100_ = lean_box(0);
return v___x_100_;
}
else
{
lean_object* v_head_101_; lean_object* v_tail_102_; lean_object* v___x_104_; uint8_t v_isShared_105_; uint8_t v_isSharedCheck_110_; 
v_head_101_ = lean_ctor_get(v_x_99_, 0);
v_tail_102_ = lean_ctor_get(v_x_99_, 1);
v_isSharedCheck_110_ = !lean_is_exclusive(v_x_99_);
if (v_isSharedCheck_110_ == 0)
{
v___x_104_ = v_x_99_;
v_isShared_105_ = v_isSharedCheck_110_;
goto v_resetjp_103_;
}
else
{
lean_inc(v_tail_102_);
lean_inc(v_head_101_);
lean_dec(v_x_99_);
v___x_104_ = lean_box(0);
v_isShared_105_ = v_isSharedCheck_110_;
goto v_resetjp_103_;
}
v_resetjp_103_:
{
lean_object* v___x_107_; 
if (v_isShared_105_ == 0)
{
lean_ctor_set_tag(v___x_104_, 0);
v___x_107_ = v___x_104_;
goto v_reusejp_106_;
}
else
{
lean_object* v_reuseFailAlloc_109_; 
v_reuseFailAlloc_109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_109_, 0, v_head_101_);
lean_ctor_set(v_reuseFailAlloc_109_, 1, v_tail_102_);
v___x_107_ = v_reuseFailAlloc_109_;
goto v_reusejp_106_;
}
v_reusejp_106_:
{
lean_object* v___x_108_; 
v___x_108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
return v___x_108_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_next_x3f(lean_object* v_00_u03b1_111_, lean_object* v_x_112_){
_start:
{
if (lean_obj_tag(v_x_112_) == 0)
{
lean_object* v___x_113_; 
v___x_113_ = lean_box(0);
return v___x_113_;
}
else
{
lean_object* v_head_114_; lean_object* v_tail_115_; lean_object* v___x_117_; uint8_t v_isShared_118_; uint8_t v_isSharedCheck_123_; 
v_head_114_ = lean_ctor_get(v_x_112_, 0);
v_tail_115_ = lean_ctor_get(v_x_112_, 1);
v_isSharedCheck_123_ = !lean_is_exclusive(v_x_112_);
if (v_isSharedCheck_123_ == 0)
{
v___x_117_ = v_x_112_;
v_isShared_118_ = v_isSharedCheck_123_;
goto v_resetjp_116_;
}
else
{
lean_inc(v_tail_115_);
lean_inc(v_head_114_);
lean_dec(v_x_112_);
v___x_117_ = lean_box(0);
v_isShared_118_ = v_isSharedCheck_123_;
goto v_resetjp_116_;
}
v_resetjp_116_:
{
lean_object* v___x_120_; 
if (v_isShared_118_ == 0)
{
lean_ctor_set_tag(v___x_117_, 0);
v___x_120_ = v___x_117_;
goto v_reusejp_119_;
}
else
{
lean_object* v_reuseFailAlloc_122_; 
v_reuseFailAlloc_122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_122_, 0, v_head_114_);
lean_ctor_set(v_reuseFailAlloc_122_, 1, v_tail_115_);
v___x_120_ = v_reuseFailAlloc_122_;
goto v_reusejp_119_;
}
v_reusejp_119_:
{
lean_object* v___x_121_; 
v___x_121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
return v___x_121_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_after___redArg(lean_object* v_p_124_, lean_object* v_x_125_){
_start:
{
if (lean_obj_tag(v_x_125_) == 0)
{
lean_dec_ref(v_p_124_);
return v_x_125_;
}
else
{
lean_object* v_head_126_; lean_object* v_tail_127_; lean_object* v___x_128_; uint8_t v___x_129_; 
v_head_126_ = lean_ctor_get(v_x_125_, 0);
lean_inc(v_head_126_);
v_tail_127_ = lean_ctor_get(v_x_125_, 1);
lean_inc(v_tail_127_);
lean_dec_ref_known(v_x_125_, 2);
lean_inc_ref(v_p_124_);
v___x_128_ = lean_apply_1(v_p_124_, v_head_126_);
v___x_129_ = lean_unbox(v___x_128_);
if (v___x_129_ == 0)
{
v_x_125_ = v_tail_127_;
goto _start;
}
else
{
lean_dec_ref(v_p_124_);
return v_tail_127_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_after(lean_object* v_00_u03b1_131_, lean_object* v_p_132_, lean_object* v_x_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lp_batteries_List_after___redArg(v_p_132_, v_x_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_replaceF___redArg(lean_object* v_f_135_, lean_object* v_x_136_){
_start:
{
if (lean_obj_tag(v_x_136_) == 0)
{
lean_dec_ref(v_f_135_);
return v_x_136_;
}
else
{
lean_object* v_head_137_; lean_object* v_tail_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_151_; 
v_head_137_ = lean_ctor_get(v_x_136_, 0);
v_tail_138_ = lean_ctor_get(v_x_136_, 1);
v_isSharedCheck_151_ = !lean_is_exclusive(v_x_136_);
if (v_isSharedCheck_151_ == 0)
{
v___x_140_ = v_x_136_;
v_isShared_141_ = v_isSharedCheck_151_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_tail_138_);
lean_inc(v_head_137_);
lean_dec(v_x_136_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_151_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_142_; 
lean_inc_ref(v_f_135_);
lean_inc(v_head_137_);
v___x_142_ = lean_apply_1(v_f_135_, v_head_137_);
if (lean_obj_tag(v___x_142_) == 0)
{
lean_object* v___x_143_; lean_object* v___x_145_; 
v___x_143_ = lp_batteries_List_replaceF___redArg(v_f_135_, v_tail_138_);
if (v_isShared_141_ == 0)
{
lean_ctor_set(v___x_140_, 1, v___x_143_);
v___x_145_ = v___x_140_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v_head_137_);
lean_ctor_set(v_reuseFailAlloc_146_, 1, v___x_143_);
v___x_145_ = v_reuseFailAlloc_146_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
return v___x_145_;
}
}
else
{
lean_object* v_val_147_; lean_object* v___x_149_; 
lean_dec(v_head_137_);
lean_dec_ref(v_f_135_);
v_val_147_ = lean_ctor_get(v___x_142_, 0);
lean_inc(v_val_147_);
lean_dec_ref_known(v___x_142_, 1);
if (v_isShared_141_ == 0)
{
lean_ctor_set(v___x_140_, 0, v_val_147_);
v___x_149_ = v___x_140_;
goto v_reusejp_148_;
}
else
{
lean_object* v_reuseFailAlloc_150_; 
v_reuseFailAlloc_150_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_150_, 0, v_val_147_);
lean_ctor_set(v_reuseFailAlloc_150_, 1, v_tail_138_);
v___x_149_ = v_reuseFailAlloc_150_;
goto v_reusejp_148_;
}
v_reusejp_148_:
{
return v___x_149_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_replaceF(lean_object* v_00_u03b1_152_, lean_object* v_f_153_, lean_object* v_x_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_batteries_List_replaceF___redArg(v_f_153_, v_x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_next_x3f_match__1_splitter___redArg(lean_object* v_x_156_, lean_object* v_h__1_157_, lean_object* v_h__2_158_){
_start:
{
if (lean_obj_tag(v_x_156_) == 0)
{
lean_object* v___x_159_; lean_object* v___x_160_; 
lean_dec(v_h__2_158_);
v___x_159_ = lean_box(0);
v___x_160_ = lean_apply_1(v_h__1_157_, v___x_159_);
return v___x_160_;
}
else
{
lean_object* v_head_161_; lean_object* v_tail_162_; lean_object* v___x_163_; 
lean_dec(v_h__1_157_);
v_head_161_ = lean_ctor_get(v_x_156_, 0);
lean_inc(v_head_161_);
v_tail_162_ = lean_ctor_get(v_x_156_, 1);
lean_inc(v_tail_162_);
lean_dec_ref_known(v_x_156_, 2);
v___x_163_ = lean_apply_2(v_h__2_158_, v_head_161_, v_tail_162_);
return v___x_163_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_next_x3f_match__1_splitter(lean_object* v_00_u03b1_164_, lean_object* v_motive_165_, lean_object* v_x_166_, lean_object* v_h__1_167_, lean_object* v_h__2_168_){
_start:
{
if (lean_obj_tag(v_x_166_) == 0)
{
lean_object* v___x_169_; lean_object* v___x_170_; 
lean_dec(v_h__2_168_);
v___x_169_ = lean_box(0);
v___x_170_ = lean_apply_1(v_h__1_167_, v___x_169_);
return v___x_170_;
}
else
{
lean_object* v_head_171_; lean_object* v_tail_172_; lean_object* v___x_173_; 
lean_dec(v_h__1_167_);
v_head_171_ = lean_ctor_get(v_x_166_, 0);
lean_inc(v_head_171_);
v_tail_172_ = lean_ctor_get(v_x_166_, 1);
lean_inc(v_tail_172_);
lean_dec_ref_known(v_x_166_, 2);
v___x_173_ = lean_apply_2(v_h__2_168_, v_head_171_, v_tail_172_);
return v___x_173_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_max_x21_match__1_splitter___redArg(lean_object* v_x_174_, lean_object* v_h__1_175_, lean_object* v_h__2_176_){
_start:
{
if (lean_obj_tag(v_x_174_) == 0)
{
lean_object* v___x_177_; lean_object* v___x_178_; 
lean_dec(v_h__2_176_);
v___x_177_ = lean_box(0);
v___x_178_ = lean_apply_1(v_h__1_175_, v___x_177_);
return v___x_178_;
}
else
{
lean_object* v_val_179_; lean_object* v___x_180_; 
lean_dec(v_h__1_175_);
v_val_179_ = lean_ctor_get(v_x_174_, 0);
lean_inc(v_val_179_);
lean_dec_ref_known(v_x_174_, 1);
v___x_180_ = lean_apply_1(v_h__2_176_, v_val_179_);
return v___x_180_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_max_x21_match__1_splitter(lean_object* v_00_u03b1_181_, lean_object* v_motive_182_, lean_object* v_x_183_, lean_object* v_h__1_184_, lean_object* v_h__2_185_){
_start:
{
if (lean_obj_tag(v_x_183_) == 0)
{
lean_object* v___x_186_; lean_object* v___x_187_; 
lean_dec(v_h__2_185_);
v___x_186_ = lean_box(0);
v___x_187_ = lean_apply_1(v_h__1_184_, v___x_186_);
return v___x_187_;
}
else
{
lean_object* v_val_188_; lean_object* v___x_189_; 
lean_dec(v_h__1_184_);
v_val_188_ = lean_ctor_get(v_x_183_, 0);
lean_inc(v_val_188_);
lean_dec_ref_known(v_x_183_, 1);
v___x_189_ = lean_apply_1(v_h__2_185_, v_val_188_);
return v___x_189_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR_go___redArg___lam__0(lean_object* v_x1_190_, lean_object* v_x2_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_192_, 0, v_x1_190_);
lean_ctor_set(v___x_192_, 1, v_x2_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR_go___redArg(lean_object* v_f_213_, lean_object* v_a_214_, lean_object* v_a_215_){
_start:
{
if (lean_obj_tag(v_a_214_) == 0)
{
lean_object* v___x_216_; 
lean_dec_ref(v_f_213_);
v___x_216_ = lean_array_to_list(v_a_215_);
return v___x_216_;
}
else
{
lean_object* v_head_217_; lean_object* v_tail_218_; lean_object* v___x_220_; uint8_t v_isShared_221_; uint8_t v_isSharedCheck_237_; 
v_head_217_ = lean_ctor_get(v_a_214_, 0);
v_tail_218_ = lean_ctor_get(v_a_214_, 1);
v_isSharedCheck_237_ = !lean_is_exclusive(v_a_214_);
if (v_isSharedCheck_237_ == 0)
{
v___x_220_ = v_a_214_;
v_isShared_221_ = v_isSharedCheck_237_;
goto v_resetjp_219_;
}
else
{
lean_inc(v_tail_218_);
lean_inc(v_head_217_);
lean_dec(v_a_214_);
v___x_220_ = lean_box(0);
v_isShared_221_ = v_isSharedCheck_237_;
goto v_resetjp_219_;
}
v_resetjp_219_:
{
lean_object* v___x_222_; 
lean_inc_ref(v_f_213_);
lean_inc(v_head_217_);
v___x_222_ = lean_apply_1(v_f_213_, v_head_217_);
if (lean_obj_tag(v___x_222_) == 0)
{
lean_object* v___x_223_; 
lean_del_object(v___x_220_);
v___x_223_ = lean_array_push(v_a_215_, v_head_217_);
v_a_214_ = v_tail_218_;
v_a_215_ = v___x_223_;
goto _start;
}
else
{
lean_object* v_val_225_; lean_object* v___x_227_; 
lean_dec(v_head_217_);
lean_dec_ref(v_f_213_);
v_val_225_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_val_225_);
lean_dec_ref_known(v___x_222_, 1);
if (v_isShared_221_ == 0)
{
lean_ctor_set(v___x_220_, 0, v_val_225_);
v___x_227_ = v___x_220_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_236_; 
v_reuseFailAlloc_236_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_236_, 0, v_val_225_);
lean_ctor_set(v_reuseFailAlloc_236_, 1, v_tail_218_);
v___x_227_ = v_reuseFailAlloc_236_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; uint8_t v___x_231_; 
v___x_228_ = lean_array_get_size(v_a_215_);
v___x_229_ = lean_unsigned_to_nat(0u);
v___x_230_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__9));
v___x_231_ = lean_nat_dec_lt(v___x_229_, v___x_228_);
if (v___x_231_ == 0)
{
lean_dec_ref(v_a_215_);
return v___x_227_;
}
else
{
lean_object* v___f_232_; size_t v___x_233_; size_t v___x_234_; lean_object* v___x_235_; 
v___f_232_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__10));
v___x_233_ = lean_usize_of_nat(v___x_228_);
v___x_234_ = ((size_t)0ULL);
v___x_235_ = l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_230_, v___f_232_, v_a_215_, v___x_233_, v___x_234_, v___x_227_);
return v___x_235_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR_go(lean_object* v_00_u03b1_238_, lean_object* v_f_239_, lean_object* v_a_240_, lean_object* v_a_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_batteries_List_replaceFTR_go___redArg(v_f_239_, v_a_240_, v_a_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR___redArg(lean_object* v_f_243_, lean_object* v_l_244_){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_245_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_246_ = lp_batteries_List_replaceFTR_go___redArg(v_f_243_, v_l_244_, v___x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_replaceFTR(lean_object* v_00_u03b1_247_, lean_object* v_f_248_, lean_object* v_l_249_){
_start:
{
lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_250_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_251_ = lp_batteries_List_replaceFTR_go___redArg(v_f_248_, v_l_249_, v___x_250_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_replaceFTR_go_match__1_splitter___redArg(lean_object* v_x_252_, lean_object* v_x_253_, lean_object* v_h__1_254_, lean_object* v_h__2_255_){
_start:
{
if (lean_obj_tag(v_x_252_) == 0)
{
lean_object* v___x_256_; 
lean_dec(v_h__2_255_);
v___x_256_ = lean_apply_1(v_h__1_254_, v_x_253_);
return v___x_256_;
}
else
{
lean_object* v_head_257_; lean_object* v_tail_258_; lean_object* v___x_259_; 
lean_dec(v_h__1_254_);
v_head_257_ = lean_ctor_get(v_x_252_, 0);
lean_inc(v_head_257_);
v_tail_258_ = lean_ctor_get(v_x_252_, 1);
lean_inc(v_tail_258_);
lean_dec_ref_known(v_x_252_, 2);
v___x_259_ = lean_apply_3(v_h__2_255_, v_head_257_, v_tail_258_, v_x_253_);
return v___x_259_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_replaceFTR_go_match__1_splitter(lean_object* v_00_u03b1_260_, lean_object* v_motive_261_, lean_object* v_x_262_, lean_object* v_x_263_, lean_object* v_h__1_264_, lean_object* v_h__2_265_){
_start:
{
if (lean_obj_tag(v_x_262_) == 0)
{
lean_object* v___x_266_; 
lean_dec(v_h__2_265_);
v___x_266_ = lean_apply_1(v_h__1_264_, v_x_263_);
return v___x_266_;
}
else
{
lean_object* v_head_267_; lean_object* v_tail_268_; lean_object* v___x_269_; 
lean_dec(v_h__1_264_);
v_head_267_ = lean_ctor_get(v_x_262_, 0);
lean_inc(v_head_267_);
v_tail_268_ = lean_ctor_get(v_x_262_, 1);
lean_inc(v_tail_268_);
lean_dec_ref_known(v_x_262_, 2);
v___x_269_ = lean_apply_3(v_h__2_265_, v_head_267_, v_tail_268_, v_x_263_);
return v___x_269_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_union___redArg(lean_object* v_inst_270_, lean_object* v_l_u2081_271_, lean_object* v_l_u2082_272_){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = lean_alloc_closure((void*)(l_List_insert), 4, 2);
lean_closure_set(v___x_273_, 0, lean_box(0));
lean_closure_set(v___x_273_, 1, v_inst_270_);
v___x_274_ = l_List_foldrTR___redArg(v___x_273_, v_l_u2082_272_, v_l_u2081_271_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_union(lean_object* v_00_u03b1_275_, lean_object* v_inst_276_, lean_object* v_l_u2081_277_, lean_object* v_l_u2082_278_){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_279_ = lean_alloc_closure((void*)(l_List_insert), 4, 2);
lean_closure_set(v___x_279_, 0, lean_box(0));
lean_closure_set(v___x_279_, 1, v_inst_276_);
v___x_280_ = l_List_foldrTR___redArg(v___x_279_, v_l_u2082_278_, v_l_u2081_277_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instUnionOfBEq__batteries___redArg(lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lean_alloc_closure((void*)(lp_batteries_List_union), 4, 2);
lean_closure_set(v___x_282_, 0, lean_box(0));
lean_closure_set(v___x_282_, 1, v_inst_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instUnionOfBEq__batteries(lean_object* v_00_u03b1_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lean_alloc_closure((void*)(lp_batteries_List_union), 4, 2);
lean_closure_set(v___x_285_, 0, lean_box(0));
lean_closure_set(v___x_285_, 1, v_inst_284_);
return v___x_285_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_inter___redArg___lam__0(lean_object* v_inst_286_, lean_object* v_l_u2082_287_, lean_object* v_x_288_){
_start:
{
uint8_t v___x_289_; 
v___x_289_ = l_List_elem___redArg(v_inst_286_, v_x_288_, v_l_u2082_287_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_inter___redArg___lam__0___boxed(lean_object* v_inst_290_, lean_object* v_l_u2082_291_, lean_object* v_x_292_){
_start:
{
uint8_t v_res_293_; lean_object* v_r_294_; 
v_res_293_ = lp_batteries_List_inter___redArg___lam__0(v_inst_290_, v_l_u2082_291_, v_x_292_);
v_r_294_ = lean_box(v_res_293_);
return v_r_294_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_inter___redArg(lean_object* v_inst_295_, lean_object* v_l_u2081_296_, lean_object* v_l_u2082_297_){
_start:
{
lean_object* v___f_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___f_298_ = lean_alloc_closure((void*)(lp_batteries_List_inter___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_298_, 0, v_inst_295_);
lean_closure_set(v___f_298_, 1, v_l_u2082_297_);
v___x_299_ = lean_box(0);
v___x_300_ = l_List_filterTR_loop___redArg(v___f_298_, v_l_u2081_296_, v___x_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_inter(lean_object* v_00_u03b1_301_, lean_object* v_inst_302_, lean_object* v_l_u2081_303_, lean_object* v_l_u2082_304_){
_start:
{
lean_object* v___f_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___f_305_ = lean_alloc_closure((void*)(lp_batteries_List_inter___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_305_, 0, v_inst_302_);
lean_closure_set(v___f_305_, 1, v_l_u2082_304_);
v___x_306_ = lean_box(0);
v___x_307_ = l_List_filterTR_loop___redArg(v___f_305_, v_l_u2081_303_, v___x_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instInterOfBEq__batteries___redArg(lean_object* v_inst_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lean_alloc_closure((void*)(lp_batteries_List_inter), 4, 2);
lean_closure_set(v___x_309_, 0, lean_box(0));
lean_closure_set(v___x_309_, 1, v_inst_308_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instInterOfBEq__batteries(lean_object* v_00_u03b1_310_, lean_object* v_inst_311_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lean_alloc_closure((void*)(lp_batteries_List_inter), 4, 2);
lean_closure_set(v___x_312_, 0, lean_box(0));
lean_closure_set(v___x_312_, 1, v_inst_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_splitAtD_go___redArg(lean_object* v_dflt_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_){
_start:
{
lean_object* v_zero_317_; uint8_t v_isZero_318_; 
v_zero_317_ = lean_unsigned_to_nat(0u);
v_isZero_318_ = lean_nat_dec_eq(v_a_314_, v_zero_317_);
if (v_isZero_318_ == 1)
{
lean_object* v___x_319_; lean_object* v___x_320_; 
lean_dec(v_a_314_);
lean_dec(v_dflt_313_);
v___x_319_ = l_List_reverse___redArg(v_a_316_);
v___x_320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_320_, 0, v___x_319_);
lean_ctor_set(v___x_320_, 1, v_a_315_);
return v___x_320_;
}
else
{
if (lean_obj_tag(v_a_315_) == 0)
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
v___x_321_ = l_List_replicateTR___redArg(v_a_314_, v_dflt_313_);
v___x_322_ = l_List_reverseAux___redArg(v_a_316_, v___x_321_);
v___x_323_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_323_, 0, v___x_322_);
lean_ctor_set(v___x_323_, 1, v_a_315_);
return v___x_323_;
}
else
{
lean_object* v_head_324_; lean_object* v_tail_325_; lean_object* v___x_327_; uint8_t v_isShared_328_; uint8_t v_isSharedCheck_335_; 
v_head_324_ = lean_ctor_get(v_a_315_, 0);
v_tail_325_ = lean_ctor_get(v_a_315_, 1);
v_isSharedCheck_335_ = !lean_is_exclusive(v_a_315_);
if (v_isSharedCheck_335_ == 0)
{
v___x_327_ = v_a_315_;
v_isShared_328_ = v_isSharedCheck_335_;
goto v_resetjp_326_;
}
else
{
lean_inc(v_tail_325_);
lean_inc(v_head_324_);
lean_dec(v_a_315_);
v___x_327_ = lean_box(0);
v_isShared_328_ = v_isSharedCheck_335_;
goto v_resetjp_326_;
}
v_resetjp_326_:
{
lean_object* v_one_329_; lean_object* v_n_330_; lean_object* v___x_332_; 
v_one_329_ = lean_unsigned_to_nat(1u);
v_n_330_ = lean_nat_sub(v_a_314_, v_one_329_);
lean_dec(v_a_314_);
if (v_isShared_328_ == 0)
{
lean_ctor_set(v___x_327_, 1, v_a_316_);
v___x_332_ = v___x_327_;
goto v_reusejp_331_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v_head_324_);
lean_ctor_set(v_reuseFailAlloc_334_, 1, v_a_316_);
v___x_332_ = v_reuseFailAlloc_334_;
goto v_reusejp_331_;
}
v_reusejp_331_:
{
v_a_314_ = v_n_330_;
v_a_315_ = v_tail_325_;
v_a_316_ = v___x_332_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_splitAtD_go(lean_object* v_00_u03b1_336_, lean_object* v_dflt_337_, lean_object* v_a_338_, lean_object* v_a_339_, lean_object* v_a_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lp_batteries_List_splitAtD_go___redArg(v_dflt_337_, v_a_338_, v_a_339_, v_a_340_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_splitAtD___redArg(lean_object* v_n_342_, lean_object* v_l_343_, lean_object* v_dflt_344_){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_345_ = lean_box(0);
v___x_346_ = lp_batteries_List_splitAtD_go___redArg(v_dflt_344_, v_n_342_, v_l_343_, v___x_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_splitAtD(lean_object* v_00_u03b1_347_, lean_object* v_n_348_, lean_object* v_l_349_, lean_object* v_dflt_350_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lp_batteries_List_splitAtD___redArg(v_n_348_, v_l_349_, v_dflt_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_modifyLast_go___redArg(lean_object* v_f_352_, lean_object* v_a_353_, lean_object* v_a_354_){
_start:
{
if (lean_obj_tag(v_a_353_) == 0)
{
lean_dec_ref(v_a_354_);
lean_dec(v_f_352_);
return v_a_353_;
}
else
{
lean_object* v_tail_355_; 
v_tail_355_ = lean_ctor_get(v_a_353_, 1);
lean_inc(v_tail_355_);
if (lean_obj_tag(v_tail_355_) == 0)
{
lean_object* v_head_356_; lean_object* v___x_358_; uint8_t v_isShared_359_; uint8_t v_isSharedCheck_372_; 
v_head_356_ = lean_ctor_get(v_a_353_, 0);
v_isSharedCheck_372_ = !lean_is_exclusive(v_a_353_);
if (v_isSharedCheck_372_ == 0)
{
lean_object* v_unused_373_; 
v_unused_373_ = lean_ctor_get(v_a_353_, 1);
lean_dec(v_unused_373_);
v___x_358_ = v_a_353_;
v_isShared_359_ = v_isSharedCheck_372_;
goto v_resetjp_357_;
}
else
{
lean_inc(v_head_356_);
lean_dec(v_a_353_);
v___x_358_ = lean_box(0);
v_isShared_359_ = v_isSharedCheck_372_;
goto v_resetjp_357_;
}
v_resetjp_357_:
{
lean_object* v___x_360_; lean_object* v___x_362_; 
v___x_360_ = lean_apply_1(v_f_352_, v_head_356_);
if (v_isShared_359_ == 0)
{
lean_ctor_set(v___x_358_, 0, v___x_360_);
v___x_362_ = v___x_358_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_371_; 
v_reuseFailAlloc_371_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_371_, 0, v___x_360_);
lean_ctor_set(v_reuseFailAlloc_371_, 1, v_tail_355_);
v___x_362_ = v_reuseFailAlloc_371_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; uint8_t v___x_366_; 
v___x_363_ = lean_array_get_size(v_a_354_);
v___x_364_ = lean_unsigned_to_nat(0u);
v___x_365_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__9));
v___x_366_ = lean_nat_dec_lt(v___x_364_, v___x_363_);
if (v___x_366_ == 0)
{
lean_dec_ref(v_a_354_);
return v___x_362_;
}
else
{
lean_object* v___f_367_; size_t v___x_368_; size_t v___x_369_; lean_object* v___x_370_; 
v___f_367_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__10));
v___x_368_ = lean_usize_of_nat(v___x_363_);
v___x_369_ = ((size_t)0ULL);
v___x_370_ = l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_365_, v___f_367_, v_a_354_, v___x_368_, v___x_369_, v___x_362_);
return v___x_370_;
}
}
}
}
else
{
lean_object* v_head_374_; lean_object* v___x_375_; 
v_head_374_ = lean_ctor_get(v_a_353_, 0);
lean_inc(v_head_374_);
lean_dec_ref_known(v_a_353_, 2);
v___x_375_ = lean_array_push(v_a_354_, v_head_374_);
v_a_353_ = v_tail_355_;
v_a_354_ = v___x_375_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_modifyLast_go(lean_object* v_00_u03b1_377_, lean_object* v_f_378_, lean_object* v_a_379_, lean_object* v_a_380_){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lp_batteries_List_modifyLast_go___redArg(v_f_378_, v_a_379_, v_a_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_modifyLast___redArg(lean_object* v_f_382_, lean_object* v_l_383_){
_start:
{
lean_object* v___x_384_; lean_object* v___x_385_; 
v___x_384_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_385_ = lp_batteries_List_modifyLast_go___redArg(v_f_382_, v_l_383_, v___x_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_modifyLast(lean_object* v_00_u03b1_386_, lean_object* v_f_387_, lean_object* v_l_388_){
_start:
{
lean_object* v___x_389_; lean_object* v___x_390_; 
v___x_389_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_390_ = lp_batteries_List_modifyLast_go___redArg(v_f_387_, v_l_388_, v___x_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeD___redArg(lean_object* v_x_391_, lean_object* v_x_392_, lean_object* v_x_393_){
_start:
{
lean_object* v_zero_394_; uint8_t v_isZero_395_; 
v_zero_394_ = lean_unsigned_to_nat(0u);
v_isZero_395_ = lean_nat_dec_eq(v_x_391_, v_zero_394_);
if (v_isZero_395_ == 1)
{
lean_object* v___x_396_; 
lean_dec(v_x_393_);
lean_dec(v_x_392_);
v___x_396_ = lean_box(0);
return v___x_396_;
}
else
{
lean_object* v_one_397_; lean_object* v_n_398_; lean_object* v___y_400_; lean_object* v___y_401_; lean_object* v___y_405_; 
v_one_397_ = lean_unsigned_to_nat(1u);
v_n_398_ = lean_nat_sub(v_x_391_, v_one_397_);
if (lean_obj_tag(v_x_392_) == 0)
{
lean_inc(v_x_393_);
v___y_405_ = v_x_393_;
goto v___jp_404_;
}
else
{
lean_object* v_head_407_; 
v_head_407_ = lean_ctor_get(v_x_392_, 0);
lean_inc(v_head_407_);
v___y_405_ = v_head_407_;
goto v___jp_404_;
}
v___jp_399_:
{
lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_402_ = lp_batteries_List_takeD___redArg(v_n_398_, v___y_401_, v_x_393_);
lean_dec(v_n_398_);
v___x_403_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_403_, 0, v___y_400_);
lean_ctor_set(v___x_403_, 1, v___x_402_);
return v___x_403_;
}
v___jp_404_:
{
if (lean_obj_tag(v_x_392_) == 0)
{
v___y_400_ = v___y_405_;
v___y_401_ = v_x_392_;
goto v___jp_399_;
}
else
{
lean_object* v_tail_406_; 
v_tail_406_ = lean_ctor_get(v_x_392_, 1);
lean_inc(v_tail_406_);
lean_dec_ref_known(v_x_392_, 2);
v___y_400_ = v___y_405_;
v___y_401_ = v_tail_406_;
goto v___jp_399_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeD___redArg___boxed(lean_object* v_x_408_, lean_object* v_x_409_, lean_object* v_x_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_batteries_List_takeD___redArg(v_x_408_, v_x_409_, v_x_410_);
lean_dec(v_x_408_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeD(lean_object* v_00_u03b1_412_, lean_object* v_x_413_, lean_object* v_x_414_, lean_object* v_x_415_){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lp_batteries_List_takeD___redArg(v_x_413_, v_x_414_, v_x_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeD___boxed(lean_object* v_00_u03b1_417_, lean_object* v_x_418_, lean_object* v_x_419_, lean_object* v_x_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_batteries_List_takeD(v_00_u03b1_417_, v_x_418_, v_x_419_, v_x_420_);
lean_dec(v_x_418_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter___redArg(lean_object* v_x_422_, lean_object* v_x_423_, lean_object* v_x_424_, lean_object* v_h__1_425_, lean_object* v_h__2_426_){
_start:
{
lean_object* v_zero_427_; uint8_t v_isZero_428_; 
v_zero_427_ = lean_unsigned_to_nat(0u);
v_isZero_428_ = lean_nat_dec_eq(v_x_422_, v_zero_427_);
if (v_isZero_428_ == 1)
{
lean_object* v___x_429_; 
lean_dec(v_h__2_426_);
v___x_429_ = lean_apply_2(v_h__1_425_, v_x_423_, v_x_424_);
return v___x_429_;
}
else
{
lean_object* v_one_430_; lean_object* v_n_431_; lean_object* v___x_432_; 
lean_dec(v_h__1_425_);
v_one_430_ = lean_unsigned_to_nat(1u);
v_n_431_ = lean_nat_sub(v_x_422_, v_one_430_);
v___x_432_ = lean_apply_3(v_h__2_426_, v_n_431_, v_x_423_, v_x_424_);
return v___x_432_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter___redArg___boxed(lean_object* v_x_433_, lean_object* v_x_434_, lean_object* v_x_435_, lean_object* v_h__1_436_, lean_object* v_h__2_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter___redArg(v_x_433_, v_x_434_, v_x_435_, v_h__1_436_, v_h__2_437_);
lean_dec(v_x_433_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter(lean_object* v_00_u03b1_439_, lean_object* v_motive_440_, lean_object* v_x_441_, lean_object* v_x_442_, lean_object* v_x_443_, lean_object* v_h__1_444_, lean_object* v_h__2_445_){
_start:
{
lean_object* v_zero_446_; uint8_t v_isZero_447_; 
v_zero_446_ = lean_unsigned_to_nat(0u);
v_isZero_447_ = lean_nat_dec_eq(v_x_441_, v_zero_446_);
if (v_isZero_447_ == 1)
{
lean_object* v___x_448_; 
lean_dec(v_h__2_445_);
v___x_448_ = lean_apply_2(v_h__1_444_, v_x_442_, v_x_443_);
return v___x_448_;
}
else
{
lean_object* v_one_449_; lean_object* v_n_450_; lean_object* v___x_451_; 
lean_dec(v_h__1_444_);
v_one_449_ = lean_unsigned_to_nat(1u);
v_n_450_ = lean_nat_sub(v_x_441_, v_one_449_);
v___x_451_ = lean_apply_3(v_h__2_445_, v_n_450_, v_x_442_, v_x_443_);
return v___x_451_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter___boxed(lean_object* v_00_u03b1_452_, lean_object* v_motive_453_, lean_object* v_x_454_, lean_object* v_x_455_, lean_object* v_x_456_, lean_object* v_h__1_457_, lean_object* v_h__2_458_){
_start:
{
lean_object* v_res_459_; 
v_res_459_ = lp_batteries___private_Batteries_Data_List_Basic_0__List_takeD_match__1_splitter(v_00_u03b1_452_, v_motive_453_, v_x_454_, v_x_455_, v_x_456_, v_h__1_457_, v_h__2_458_);
lean_dec(v_x_454_);
return v_res_459_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg(lean_object* v_as_460_, size_t v_i_461_, size_t v_stop_462_, lean_object* v_b_463_){
_start:
{
uint8_t v___x_464_; 
v___x_464_ = lean_usize_dec_eq(v_i_461_, v_stop_462_);
if (v___x_464_ == 0)
{
size_t v___x_465_; size_t v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; 
v___x_465_ = ((size_t)1ULL);
v___x_466_ = lean_usize_sub(v_i_461_, v___x_465_);
v___x_467_ = lean_array_uget_borrowed(v_as_460_, v___x_466_);
lean_inc(v___x_467_);
v___x_468_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_468_, 0, v___x_467_);
lean_ctor_set(v___x_468_, 1, v_b_463_);
v_i_461_ = v___x_466_;
v_b_463_ = v___x_468_;
goto _start;
}
else
{
return v_b_463_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg___boxed(lean_object* v_as_470_, lean_object* v_i_471_, lean_object* v_stop_472_, lean_object* v_b_473_){
_start:
{
size_t v_i_boxed_474_; size_t v_stop_boxed_475_; lean_object* v_res_476_; 
v_i_boxed_474_ = lean_unbox_usize(v_i_471_);
lean_dec(v_i_471_);
v_stop_boxed_475_ = lean_unbox_usize(v_stop_472_);
lean_dec(v_stop_472_);
v_res_476_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg(v_as_470_, v_i_boxed_474_, v_stop_boxed_475_, v_b_473_);
lean_dec_ref(v_as_470_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeDTR_go___redArg(lean_object* v_dflt_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_){
_start:
{
lean_object* v_zero_481_; uint8_t v_isZero_482_; 
v_zero_481_ = lean_unsigned_to_nat(0u);
v_isZero_482_ = lean_nat_dec_eq(v_a_478_, v_zero_481_);
if (v_isZero_482_ == 1)
{
lean_object* v___x_483_; 
lean_dec(v_a_479_);
lean_dec(v_a_478_);
lean_dec(v_dflt_477_);
v___x_483_ = lean_array_to_list(v_a_480_);
return v___x_483_;
}
else
{
if (lean_obj_tag(v_a_479_) == 0)
{
lean_object* v___x_484_; lean_object* v___x_485_; uint8_t v___x_486_; 
v___x_484_ = l_List_replicateTR___redArg(v_a_478_, v_dflt_477_);
v___x_485_ = lean_array_get_size(v_a_480_);
v___x_486_ = lean_nat_dec_lt(v_zero_481_, v___x_485_);
if (v___x_486_ == 0)
{
lean_dec_ref(v_a_480_);
return v___x_484_;
}
else
{
size_t v___x_487_; size_t v___x_488_; lean_object* v___x_489_; 
v___x_487_ = lean_usize_of_nat(v___x_485_);
v___x_488_ = ((size_t)0ULL);
v___x_489_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg(v_a_480_, v___x_487_, v___x_488_, v___x_484_);
lean_dec_ref(v_a_480_);
return v___x_489_;
}
}
else
{
lean_object* v_head_490_; lean_object* v_tail_491_; lean_object* v_one_492_; lean_object* v_n_493_; lean_object* v___x_494_; 
v_head_490_ = lean_ctor_get(v_a_479_, 0);
lean_inc(v_head_490_);
v_tail_491_ = lean_ctor_get(v_a_479_, 1);
lean_inc(v_tail_491_);
lean_dec_ref_known(v_a_479_, 2);
v_one_492_ = lean_unsigned_to_nat(1u);
v_n_493_ = lean_nat_sub(v_a_478_, v_one_492_);
lean_dec(v_a_478_);
v___x_494_ = lean_array_push(v_a_480_, v_head_490_);
v_a_478_ = v_n_493_;
v_a_479_ = v_tail_491_;
v_a_480_ = v___x_494_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeDTR_go(lean_object* v_00_u03b1_496_, lean_object* v_dflt_497_, lean_object* v_a_498_, lean_object* v_a_499_, lean_object* v_a_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lp_batteries_List_takeDTR_go___redArg(v_dflt_497_, v_a_498_, v_a_499_, v_a_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0(lean_object* v_00_u03b1_502_, lean_object* v_as_503_, size_t v_i_504_, size_t v_stop_505_, lean_object* v_b_506_){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg(v_as_503_, v_i_504_, v_stop_505_, v_b_506_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___boxed(lean_object* v_00_u03b1_508_, lean_object* v_as_509_, lean_object* v_i_510_, lean_object* v_stop_511_, lean_object* v_b_512_){
_start:
{
size_t v_i_boxed_513_; size_t v_stop_boxed_514_; lean_object* v_res_515_; 
v_i_boxed_513_ = lean_unbox_usize(v_i_510_);
lean_dec(v_i_510_);
v_stop_boxed_514_ = lean_unbox_usize(v_stop_511_);
lean_dec(v_stop_511_);
v_res_515_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0(v_00_u03b1_508_, v_as_509_, v_i_boxed_513_, v_stop_boxed_514_, v_b_512_);
lean_dec_ref(v_as_509_);
return v_res_515_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeDTR___redArg(lean_object* v_n_516_, lean_object* v_l_517_, lean_object* v_dflt_518_){
_start:
{
lean_object* v___x_519_; lean_object* v___x_520_; 
v___x_519_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_520_ = lp_batteries_List_takeDTR_go___redArg(v_dflt_518_, v_n_516_, v_l_517_, v___x_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeDTR(lean_object* v_00_u03b1_521_, lean_object* v_n_522_, lean_object* v_l_523_, lean_object* v_dflt_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_batteries_List_takeDTR___redArg(v_n_522_, v_l_523_, v_dflt_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeDTR_go_match__1_splitter___redArg(lean_object* v_x_526_, lean_object* v_x_527_, lean_object* v_x_528_, lean_object* v_h__1_529_, lean_object* v_h__2_530_, lean_object* v_h__3_531_){
_start:
{
lean_object* v_zero_532_; uint8_t v_isZero_533_; 
v_zero_532_ = lean_unsigned_to_nat(0u);
v_isZero_533_ = lean_nat_dec_eq(v_x_526_, v_zero_532_);
if (v_isZero_533_ == 1)
{
lean_object* v___x_534_; 
lean_dec(v_h__3_531_);
lean_dec(v_h__1_529_);
lean_dec(v_x_526_);
v___x_534_ = lean_apply_2(v_h__2_530_, v_x_527_, v_x_528_);
return v___x_534_;
}
else
{
lean_dec(v_h__2_530_);
if (lean_obj_tag(v_x_527_) == 0)
{
lean_object* v___x_535_; 
lean_dec(v_h__1_529_);
v___x_535_ = lean_apply_3(v_h__3_531_, v_x_526_, v_x_528_, lean_box(0));
return v___x_535_;
}
else
{
lean_object* v_head_536_; lean_object* v_tail_537_; lean_object* v_one_538_; lean_object* v_n_539_; lean_object* v___x_540_; 
lean_dec(v_h__3_531_);
v_head_536_ = lean_ctor_get(v_x_527_, 0);
lean_inc(v_head_536_);
v_tail_537_ = lean_ctor_get(v_x_527_, 1);
lean_inc(v_tail_537_);
lean_dec_ref_known(v_x_527_, 2);
v_one_538_ = lean_unsigned_to_nat(1u);
v_n_539_ = lean_nat_sub(v_x_526_, v_one_538_);
lean_dec(v_x_526_);
v___x_540_ = lean_apply_4(v_h__1_529_, v_n_539_, v_head_536_, v_tail_537_, v_x_528_);
return v___x_540_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeDTR_go_match__1_splitter(lean_object* v_00_u03b1_541_, lean_object* v_motive_542_, lean_object* v_x_543_, lean_object* v_x_544_, lean_object* v_x_545_, lean_object* v_h__1_546_, lean_object* v_h__2_547_, lean_object* v_h__3_548_){
_start:
{
lean_object* v_zero_549_; uint8_t v_isZero_550_; 
v_zero_549_ = lean_unsigned_to_nat(0u);
v_isZero_550_ = lean_nat_dec_eq(v_x_543_, v_zero_549_);
if (v_isZero_550_ == 1)
{
lean_object* v___x_551_; 
lean_dec(v_h__3_548_);
lean_dec(v_h__1_546_);
lean_dec(v_x_543_);
v___x_551_ = lean_apply_2(v_h__2_547_, v_x_544_, v_x_545_);
return v___x_551_;
}
else
{
lean_dec(v_h__2_547_);
if (lean_obj_tag(v_x_544_) == 0)
{
lean_object* v___x_552_; 
lean_dec(v_h__1_546_);
v___x_552_ = lean_apply_3(v_h__3_548_, v_x_543_, v_x_545_, lean_box(0));
return v___x_552_;
}
else
{
lean_object* v_head_553_; lean_object* v_tail_554_; lean_object* v_one_555_; lean_object* v_n_556_; lean_object* v___x_557_; 
lean_dec(v_h__3_548_);
v_head_553_ = lean_ctor_get(v_x_544_, 0);
lean_inc(v_head_553_);
v_tail_554_ = lean_ctor_get(v_x_544_, 1);
lean_inc(v_tail_554_);
lean_dec_ref_known(v_x_544_, 2);
v_one_555_ = lean_unsigned_to_nat(1u);
v_n_556_ = lean_nat_sub(v_x_543_, v_one_555_);
lean_dec(v_x_543_);
v___x_557_ = lean_apply_4(v_h__1_546_, v_n_556_, v_head_553_, v_tail_554_, v_x_545_);
return v___x_557_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM_go___redArg(lean_object* v_inst_558_, lean_object* v_f_559_, lean_object* v_a_560_, lean_object* v_a_561_, lean_object* v_a_562_){
_start:
{
if (lean_obj_tag(v_a_560_) == 0)
{
lean_object* v_toApplicative_563_; lean_object* v___x_565_; uint8_t v_isShared_566_; uint8_t v_isSharedCheck_572_; 
v_toApplicative_563_ = lean_ctor_get(v_inst_558_, 0);
lean_inc_ref(v_toApplicative_563_);
lean_dec(v_f_559_);
v_isSharedCheck_572_ = !lean_is_exclusive(v_inst_558_);
if (v_isSharedCheck_572_ == 0)
{
lean_object* v_unused_573_; lean_object* v_unused_574_; 
v_unused_573_ = lean_ctor_get(v_inst_558_, 1);
lean_dec(v_unused_573_);
v_unused_574_ = lean_ctor_get(v_inst_558_, 0);
lean_dec(v_unused_574_);
v___x_565_ = v_inst_558_;
v_isShared_566_ = v_isSharedCheck_572_;
goto v_resetjp_564_;
}
else
{
lean_dec(v_inst_558_);
v___x_565_ = lean_box(0);
v_isShared_566_ = v_isSharedCheck_572_;
goto v_resetjp_564_;
}
v_resetjp_564_:
{
lean_object* v_toPure_567_; lean_object* v___x_569_; 
v_toPure_567_ = lean_ctor_get(v_toApplicative_563_, 1);
lean_inc(v_toPure_567_);
lean_dec_ref(v_toApplicative_563_);
if (v_isShared_566_ == 0)
{
lean_ctor_set_tag(v___x_565_, 1);
lean_ctor_set(v___x_565_, 1, v_a_562_);
lean_ctor_set(v___x_565_, 0, v_a_561_);
v___x_569_ = v___x_565_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v_a_561_);
lean_ctor_set(v_reuseFailAlloc_571_, 1, v_a_562_);
v___x_569_ = v_reuseFailAlloc_571_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
lean_object* v___x_570_; 
v___x_570_ = lean_apply_2(v_toPure_567_, lean_box(0), v___x_569_);
return v___x_570_;
}
}
}
else
{
lean_object* v_toBind_575_; lean_object* v_head_576_; lean_object* v_tail_577_; lean_object* v___f_578_; lean_object* v___x_579_; lean_object* v___x_580_; 
v_toBind_575_ = lean_ctor_get(v_inst_558_, 1);
lean_inc(v_toBind_575_);
v_head_576_ = lean_ctor_get(v_a_560_, 0);
lean_inc(v_head_576_);
v_tail_577_ = lean_ctor_get(v_a_560_, 1);
lean_inc(v_tail_577_);
lean_dec_ref_known(v_a_560_, 2);
lean_inc(v_f_559_);
lean_inc(v_a_561_);
v___f_578_ = lean_alloc_closure((void*)(lp_batteries_List_scanAuxM_go___redArg___lam__0), 6, 5);
lean_closure_set(v___f_578_, 0, v_a_561_);
lean_closure_set(v___f_578_, 1, v_a_562_);
lean_closure_set(v___f_578_, 2, v_inst_558_);
lean_closure_set(v___f_578_, 3, v_f_559_);
lean_closure_set(v___f_578_, 4, v_tail_577_);
v___x_579_ = lean_apply_2(v_f_559_, v_a_561_, v_head_576_);
v___x_580_ = lean_apply_4(v_toBind_575_, lean_box(0), lean_box(0), v___x_579_, v___f_578_);
return v___x_580_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM_go___redArg___lam__0(lean_object* v_a_581_, lean_object* v_a_582_, lean_object* v_inst_583_, lean_object* v_f_584_, lean_object* v_tail_585_, lean_object* v_____do__lift_586_){
_start:
{
lean_object* v___x_587_; lean_object* v___x_588_; 
v___x_587_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_587_, 0, v_a_581_);
lean_ctor_set(v___x_587_, 1, v_a_582_);
v___x_588_ = lp_batteries_List_scanAuxM_go___redArg(v_inst_583_, v_f_584_, v_tail_585_, v_____do__lift_586_, v___x_587_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM_go(lean_object* v_m_589_, lean_object* v_00_u03b2_590_, lean_object* v_00_u03b1_591_, lean_object* v_inst_592_, lean_object* v_f_593_, lean_object* v_a_594_, lean_object* v_a_595_, lean_object* v_a_596_){
_start:
{
lean_object* v___x_597_; 
v___x_597_ = lp_batteries_List_scanAuxM_go___redArg(v_inst_592_, v_f_593_, v_a_594_, v_a_595_, v_a_596_);
return v___x_597_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM___redArg(lean_object* v_inst_598_, lean_object* v_f_599_, lean_object* v_init_600_, lean_object* v_l_601_){
_start:
{
lean_object* v___x_602_; lean_object* v___x_603_; 
v___x_602_ = lean_box(0);
v___x_603_ = lp_batteries_List_scanAuxM_go___redArg(v_inst_598_, v_f_599_, v_l_601_, v_init_600_, v___x_602_);
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_scanAuxM(lean_object* v_m_604_, lean_object* v_00_u03b2_605_, lean_object* v_00_u03b1_606_, lean_object* v_inst_607_, lean_object* v_f_608_, lean_object* v_init_609_, lean_object* v_l_610_){
_start:
{
lean_object* v___x_611_; lean_object* v___x_612_; 
v___x_611_ = lean_box(0);
v___x_612_ = lp_batteries_List_scanAuxM_go___redArg(v_inst_607_, v_f_608_, v_l_610_, v_init_609_, v___x_611_);
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldlIdx___redArg(lean_object* v_f_613_, lean_object* v_init_614_, lean_object* v_x_615_, lean_object* v_x_616_){
_start:
{
if (lean_obj_tag(v_x_615_) == 0)
{
lean_dec(v_x_616_);
lean_dec(v_f_613_);
return v_init_614_;
}
else
{
lean_object* v_head_617_; lean_object* v_tail_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; 
v_head_617_ = lean_ctor_get(v_x_615_, 0);
lean_inc(v_head_617_);
v_tail_618_ = lean_ctor_get(v_x_615_, 1);
lean_inc(v_tail_618_);
lean_dec_ref_known(v_x_615_, 2);
lean_inc(v_f_613_);
lean_inc(v_x_616_);
v___x_619_ = lean_apply_3(v_f_613_, v_x_616_, v_init_614_, v_head_617_);
v___x_620_ = lean_unsigned_to_nat(1u);
v___x_621_ = lean_nat_add(v_x_616_, v___x_620_);
lean_dec(v_x_616_);
v_init_614_ = v___x_619_;
v_x_615_ = v_tail_618_;
v_x_616_ = v___x_621_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldlIdx(lean_object* v_00_u03b1_623_, lean_object* v_00_u03b2_624_, lean_object* v_f_625_, lean_object* v_init_626_, lean_object* v_x_627_, lean_object* v_x_628_){
_start:
{
lean_object* v___x_629_; 
v___x_629_ = lp_batteries_List_foldlIdx___redArg(v_f_625_, v_init_626_, v_x_627_, v_x_628_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdx___redArg(lean_object* v_f_630_, lean_object* v_init_631_, lean_object* v_x_632_, lean_object* v_x_633_){
_start:
{
if (lean_obj_tag(v_x_632_) == 0)
{
lean_dec(v_x_633_);
lean_dec(v_f_630_);
lean_inc(v_init_631_);
return v_init_631_;
}
else
{
lean_object* v_head_634_; lean_object* v_tail_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; 
v_head_634_ = lean_ctor_get(v_x_632_, 0);
lean_inc(v_head_634_);
v_tail_635_ = lean_ctor_get(v_x_632_, 1);
lean_inc(v_tail_635_);
lean_dec_ref_known(v_x_632_, 2);
v___x_636_ = lean_unsigned_to_nat(1u);
v___x_637_ = lean_nat_add(v_x_633_, v___x_636_);
lean_inc(v_f_630_);
v___x_638_ = lp_batteries_List_foldrIdx___redArg(v_f_630_, v_init_631_, v_tail_635_, v___x_637_);
v___x_639_ = lean_apply_3(v_f_630_, v_x_633_, v_head_634_, v___x_638_);
return v___x_639_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdx___redArg___boxed(lean_object* v_f_640_, lean_object* v_init_641_, lean_object* v_x_642_, lean_object* v_x_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_batteries_List_foldrIdx___redArg(v_f_640_, v_init_641_, v_x_642_, v_x_643_);
lean_dec(v_init_641_);
return v_res_644_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdx(lean_object* v_00_u03b1_645_, lean_object* v_00_u03b2_646_, lean_object* v_f_647_, lean_object* v_init_648_, lean_object* v_x_649_, lean_object* v_x_650_){
_start:
{
lean_object* v___x_651_; 
v___x_651_ = lp_batteries_List_foldrIdx___redArg(v_f_647_, v_init_648_, v_x_649_, v_x_650_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdx___boxed(lean_object* v_00_u03b1_652_, lean_object* v_00_u03b2_653_, lean_object* v_f_654_, lean_object* v_init_655_, lean_object* v_x_656_, lean_object* v_x_657_){
_start:
{
lean_object* v_res_658_; 
v_res_658_ = lp_batteries_List_foldrIdx(v_00_u03b1_652_, v_00_u03b2_653_, v_f_654_, v_init_655_, v_x_656_, v_x_657_);
lean_dec(v_init_655_);
return v_res_658_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR___redArg___lam__0(lean_object* v_f_659_, lean_object* v_a_660_, lean_object* v_x_661_){
_start:
{
lean_object* v_fst_662_; lean_object* v_snd_663_; lean_object* v___x_665_; uint8_t v_isShared_666_; uint8_t v_isSharedCheck_673_; 
v_fst_662_ = lean_ctor_get(v_x_661_, 0);
v_snd_663_ = lean_ctor_get(v_x_661_, 1);
v_isSharedCheck_673_ = !lean_is_exclusive(v_x_661_);
if (v_isSharedCheck_673_ == 0)
{
v___x_665_ = v_x_661_;
v_isShared_666_ = v_isSharedCheck_673_;
goto v_resetjp_664_;
}
else
{
lean_inc(v_snd_663_);
lean_inc(v_fst_662_);
lean_dec(v_x_661_);
v___x_665_ = lean_box(0);
v_isShared_666_ = v_isSharedCheck_673_;
goto v_resetjp_664_;
}
v_resetjp_664_:
{
lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_671_; 
v___x_667_ = lean_unsigned_to_nat(1u);
v___x_668_ = lean_nat_sub(v_snd_663_, v___x_667_);
lean_dec(v_snd_663_);
lean_inc(v___x_668_);
v___x_669_ = lean_apply_3(v_f_659_, v___x_668_, v_a_660_, v_fst_662_);
if (v_isShared_666_ == 0)
{
lean_ctor_set(v___x_665_, 1, v___x_668_);
lean_ctor_set(v___x_665_, 0, v___x_669_);
v___x_671_ = v___x_665_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v___x_669_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v___x_668_);
v___x_671_ = v_reuseFailAlloc_672_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
return v___x_671_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR___redArg(lean_object* v_f_674_, lean_object* v_init_675_, lean_object* v_l_676_, lean_object* v_start_677_){
_start:
{
lean_object* v___f_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v_fst_683_; 
v___f_678_ = lean_alloc_closure((void*)(lp_batteries_List_foldrIdxTR___redArg___lam__0), 3, 1);
lean_closure_set(v___f_678_, 0, v_f_674_);
v___x_679_ = l_List_lengthTR___redArg(v_l_676_);
v___x_680_ = lean_nat_add(v_start_677_, v___x_679_);
lean_dec(v___x_679_);
v___x_681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_681_, 0, v_init_675_);
lean_ctor_set(v___x_681_, 1, v___x_680_);
v___x_682_ = l_List_foldrTR___redArg(v___f_678_, v___x_681_, v_l_676_);
v_fst_683_ = lean_ctor_get(v___x_682_, 0);
lean_inc(v_fst_683_);
lean_dec(v___x_682_);
return v_fst_683_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR___redArg___boxed(lean_object* v_f_684_, lean_object* v_init_685_, lean_object* v_l_686_, lean_object* v_start_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_batteries_List_foldrIdxTR___redArg(v_f_684_, v_init_685_, v_l_686_, v_start_687_);
lean_dec(v_start_687_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR(lean_object* v_00_u03b1_689_, lean_object* v_00_u03b2_690_, lean_object* v_f_691_, lean_object* v_init_692_, lean_object* v_l_693_, lean_object* v_start_694_){
_start:
{
lean_object* v___f_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v_fst_700_; 
v___f_695_ = lean_alloc_closure((void*)(lp_batteries_List_foldrIdxTR___redArg___lam__0), 3, 1);
lean_closure_set(v___f_695_, 0, v_f_691_);
v___x_696_ = l_List_lengthTR___redArg(v_l_693_);
v___x_697_ = lean_nat_add(v_start_694_, v___x_696_);
lean_dec(v___x_696_);
v___x_698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_698_, 0, v_init_692_);
lean_ctor_set(v___x_698_, 1, v___x_697_);
v___x_699_ = l_List_foldrTR___redArg(v___f_695_, v___x_698_, v_l_693_);
v_fst_700_ = lean_ctor_get(v___x_699_, 0);
lean_inc(v_fst_700_);
lean_dec(v___x_699_);
return v_fst_700_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrIdxTR___boxed(lean_object* v_00_u03b1_701_, lean_object* v_00_u03b2_702_, lean_object* v_f_703_, lean_object* v_init_704_, lean_object* v_l_705_, lean_object* v_start_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_batteries_List_foldrIdxTR(v_00_u03b1_701_, v_00_u03b2_702_, v_f_703_, v_init_704_, v_l_705_, v_start_706_);
lean_dec(v_start_706_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_foldlIdx_match__1_splitter___redArg(lean_object* v_x_708_, lean_object* v_x_709_, lean_object* v_h__1_710_, lean_object* v_h__2_711_){
_start:
{
if (lean_obj_tag(v_x_708_) == 0)
{
lean_object* v___x_712_; 
lean_dec(v_h__2_711_);
v___x_712_ = lean_apply_1(v_h__1_710_, v_x_709_);
return v___x_712_;
}
else
{
lean_object* v_head_713_; lean_object* v_tail_714_; lean_object* v___x_715_; 
lean_dec(v_h__1_710_);
v_head_713_ = lean_ctor_get(v_x_708_, 0);
lean_inc(v_head_713_);
v_tail_714_ = lean_ctor_get(v_x_708_, 1);
lean_inc(v_tail_714_);
lean_dec_ref_known(v_x_708_, 2);
v___x_715_ = lean_apply_3(v_h__2_711_, v_head_713_, v_tail_714_, v_x_709_);
return v___x_715_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_foldlIdx_match__1_splitter(lean_object* v_00_u03b2_716_, lean_object* v_motive_717_, lean_object* v_x_718_, lean_object* v_x_719_, lean_object* v_h__1_720_, lean_object* v_h__2_721_){
_start:
{
if (lean_obj_tag(v_x_718_) == 0)
{
lean_object* v___x_722_; 
lean_dec(v_h__2_721_);
v___x_722_ = lean_apply_1(v_h__1_720_, v_x_719_);
return v___x_722_;
}
else
{
lean_object* v_head_723_; lean_object* v_tail_724_; lean_object* v___x_725_; 
lean_dec(v_h__1_720_);
v_head_723_ = lean_ctor_get(v_x_718_, 0);
lean_inc(v_head_723_);
v_tail_724_ = lean_ctor_get(v_x_718_, 1);
lean_inc(v_tail_724_);
lean_dec_ref_known(v_x_718_, 2);
v___x_725_ = lean_apply_3(v_h__2_721_, v_head_723_, v_tail_724_, v_x_719_);
return v___x_725_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs___redArg___lam__0(lean_object* v_p_726_, lean_object* v_a_727_, lean_object* v_x_728_){
_start:
{
lean_object* v_fst_729_; lean_object* v_snd_730_; lean_object* v___x_732_; uint8_t v_isShared_733_; uint8_t v_isSharedCheck_745_; 
v_fst_729_ = lean_ctor_get(v_x_728_, 0);
v_snd_730_ = lean_ctor_get(v_x_728_, 1);
v_isSharedCheck_745_ = !lean_is_exclusive(v_x_728_);
if (v_isSharedCheck_745_ == 0)
{
v___x_732_ = v_x_728_;
v_isShared_733_ = v_isSharedCheck_745_;
goto v_resetjp_731_;
}
else
{
lean_inc(v_snd_730_);
lean_inc(v_fst_729_);
lean_dec(v_x_728_);
v___x_732_ = lean_box(0);
v_isShared_733_ = v_isSharedCheck_745_;
goto v_resetjp_731_;
}
v_resetjp_731_:
{
lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; uint8_t v___x_737_; 
v___x_734_ = lean_unsigned_to_nat(1u);
v___x_735_ = lean_nat_sub(v_snd_730_, v___x_734_);
lean_dec(v_snd_730_);
v___x_736_ = lean_apply_1(v_p_726_, v_a_727_);
v___x_737_ = lean_unbox(v___x_736_);
if (v___x_737_ == 0)
{
lean_object* v___x_739_; 
if (v_isShared_733_ == 0)
{
lean_ctor_set(v___x_732_, 1, v___x_735_);
v___x_739_ = v___x_732_;
goto v_reusejp_738_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v_fst_729_);
lean_ctor_set(v_reuseFailAlloc_740_, 1, v___x_735_);
v___x_739_ = v_reuseFailAlloc_740_;
goto v_reusejp_738_;
}
v_reusejp_738_:
{
return v___x_739_;
}
}
else
{
lean_object* v___x_741_; lean_object* v___x_743_; 
lean_inc(v___x_735_);
v___x_741_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_741_, 0, v___x_735_);
lean_ctor_set(v___x_741_, 1, v_fst_729_);
if (v_isShared_733_ == 0)
{
lean_ctor_set(v___x_732_, 1, v___x_735_);
lean_ctor_set(v___x_732_, 0, v___x_741_);
v___x_743_ = v___x_732_;
goto v_reusejp_742_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v___x_741_);
lean_ctor_set(v_reuseFailAlloc_744_, 1, v___x_735_);
v___x_743_ = v_reuseFailAlloc_744_;
goto v_reusejp_742_;
}
v_reusejp_742_:
{
return v___x_743_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs___redArg(lean_object* v_p_746_, lean_object* v_l_747_, lean_object* v_start_748_){
_start:
{
lean_object* v___f_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v_fst_755_; 
v___f_749_ = lean_alloc_closure((void*)(lp_batteries_List_findIdxs___redArg___lam__0), 3, 1);
lean_closure_set(v___f_749_, 0, v_p_746_);
v___x_750_ = lean_box(0);
v___x_751_ = l_List_lengthTR___redArg(v_l_747_);
v___x_752_ = lean_nat_add(v_start_748_, v___x_751_);
lean_dec(v___x_751_);
v___x_753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_753_, 0, v___x_750_);
lean_ctor_set(v___x_753_, 1, v___x_752_);
v___x_754_ = l_List_foldrTR___redArg(v___f_749_, v___x_753_, v_l_747_);
v_fst_755_ = lean_ctor_get(v___x_754_, 0);
lean_inc(v_fst_755_);
lean_dec(v___x_754_);
return v_fst_755_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs___redArg___boxed(lean_object* v_p_756_, lean_object* v_l_757_, lean_object* v_start_758_){
_start:
{
lean_object* v_res_759_; 
v_res_759_ = lp_batteries_List_findIdxs___redArg(v_p_756_, v_l_757_, v_start_758_);
lean_dec(v_start_758_);
return v_res_759_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs(lean_object* v_00_u03b1_760_, lean_object* v_p_761_, lean_object* v_l_762_, lean_object* v_start_763_){
_start:
{
lean_object* v___f_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v_fst_770_; 
v___f_764_ = lean_alloc_closure((void*)(lp_batteries_List_findIdxs___redArg___lam__0), 3, 1);
lean_closure_set(v___f_764_, 0, v_p_761_);
v___x_765_ = lean_box(0);
v___x_766_ = l_List_lengthTR___redArg(v_l_762_);
v___x_767_ = lean_nat_add(v_start_763_, v___x_766_);
lean_dec(v___x_766_);
v___x_768_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_768_, 0, v___x_765_);
lean_ctor_set(v___x_768_, 1, v___x_767_);
v___x_769_ = l_List_foldrTR___redArg(v___f_764_, v___x_768_, v_l_762_);
v_fst_770_ = lean_ctor_get(v___x_769_, 0);
lean_inc(v_fst_770_);
lean_dec(v___x_769_);
return v_fst_770_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxs___boxed(lean_object* v_00_u03b1_771_, lean_object* v_p_772_, lean_object* v_l_773_, lean_object* v_start_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_batteries_List_findIdxs(v_00_u03b1_771_, v_p_772_, v_l_773_, v_start_774_);
lean_dec(v_start_774_);
return v_res_775_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues___redArg___lam__0(lean_object* v_p_776_, lean_object* v_a_777_, lean_object* v_x_778_){
_start:
{
lean_object* v_fst_779_; lean_object* v_snd_780_; lean_object* v___x_782_; uint8_t v_isShared_783_; uint8_t v_isSharedCheck_796_; 
v_fst_779_ = lean_ctor_get(v_x_778_, 0);
v_snd_780_ = lean_ctor_get(v_x_778_, 1);
v_isSharedCheck_796_ = !lean_is_exclusive(v_x_778_);
if (v_isSharedCheck_796_ == 0)
{
v___x_782_ = v_x_778_;
v_isShared_783_ = v_isSharedCheck_796_;
goto v_resetjp_781_;
}
else
{
lean_inc(v_snd_780_);
lean_inc(v_fst_779_);
lean_dec(v_x_778_);
v___x_782_ = lean_box(0);
v_isShared_783_ = v_isSharedCheck_796_;
goto v_resetjp_781_;
}
v_resetjp_781_:
{
lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; uint8_t v___x_787_; 
v___x_784_ = lean_unsigned_to_nat(1u);
v___x_785_ = lean_nat_sub(v_snd_780_, v___x_784_);
lean_dec(v_snd_780_);
lean_inc(v_a_777_);
v___x_786_ = lean_apply_1(v_p_776_, v_a_777_);
v___x_787_ = lean_unbox(v___x_786_);
if (v___x_787_ == 0)
{
lean_object* v___x_789_; 
lean_dec(v_a_777_);
if (v_isShared_783_ == 0)
{
lean_ctor_set(v___x_782_, 1, v___x_785_);
v___x_789_ = v___x_782_;
goto v_reusejp_788_;
}
else
{
lean_object* v_reuseFailAlloc_790_; 
v_reuseFailAlloc_790_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_790_, 0, v_fst_779_);
lean_ctor_set(v_reuseFailAlloc_790_, 1, v___x_785_);
v___x_789_ = v_reuseFailAlloc_790_;
goto v_reusejp_788_;
}
v_reusejp_788_:
{
return v___x_789_;
}
}
else
{
lean_object* v___x_792_; 
lean_inc(v___x_785_);
if (v_isShared_783_ == 0)
{
lean_ctor_set(v___x_782_, 1, v_a_777_);
lean_ctor_set(v___x_782_, 0, v___x_785_);
v___x_792_ = v___x_782_;
goto v_reusejp_791_;
}
else
{
lean_object* v_reuseFailAlloc_795_; 
v_reuseFailAlloc_795_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_795_, 0, v___x_785_);
lean_ctor_set(v_reuseFailAlloc_795_, 1, v_a_777_);
v___x_792_ = v_reuseFailAlloc_795_;
goto v_reusejp_791_;
}
v_reusejp_791_:
{
lean_object* v___x_793_; lean_object* v___x_794_; 
v___x_793_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_793_, 0, v___x_792_);
lean_ctor_set(v___x_793_, 1, v_fst_779_);
v___x_794_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_794_, 0, v___x_793_);
lean_ctor_set(v___x_794_, 1, v___x_785_);
return v___x_794_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues___redArg(lean_object* v_p_797_, lean_object* v_l_798_, lean_object* v_start_799_){
_start:
{
lean_object* v___f_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v_fst_806_; 
v___f_800_ = lean_alloc_closure((void*)(lp_batteries_List_findIdxsValues___redArg___lam__0), 3, 1);
lean_closure_set(v___f_800_, 0, v_p_797_);
v___x_801_ = lean_box(0);
v___x_802_ = l_List_lengthTR___redArg(v_l_798_);
v___x_803_ = lean_nat_add(v_start_799_, v___x_802_);
lean_dec(v___x_802_);
v___x_804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_804_, 0, v___x_801_);
lean_ctor_set(v___x_804_, 1, v___x_803_);
v___x_805_ = l_List_foldrTR___redArg(v___f_800_, v___x_804_, v_l_798_);
v_fst_806_ = lean_ctor_get(v___x_805_, 0);
lean_inc(v_fst_806_);
lean_dec(v___x_805_);
return v_fst_806_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues___redArg___boxed(lean_object* v_p_807_, lean_object* v_l_808_, lean_object* v_start_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_batteries_List_findIdxsValues___redArg(v_p_807_, v_l_808_, v_start_809_);
lean_dec(v_start_809_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues(lean_object* v_00_u03b1_811_, lean_object* v_p_812_, lean_object* v_l_813_, lean_object* v_start_814_){
_start:
{
lean_object* v___f_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v_fst_821_; 
v___f_815_ = lean_alloc_closure((void*)(lp_batteries_List_findIdxsValues___redArg___lam__0), 3, 1);
lean_closure_set(v___f_815_, 0, v_p_812_);
v___x_816_ = lean_box(0);
v___x_817_ = l_List_lengthTR___redArg(v_l_813_);
v___x_818_ = lean_nat_add(v_start_814_, v___x_817_);
lean_dec(v___x_817_);
v___x_819_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_819_, 0, v___x_816_);
lean_ctor_set(v___x_819_, 1, v___x_818_);
v___x_820_ = l_List_foldrTR___redArg(v___f_815_, v___x_819_, v_l_813_);
v_fst_821_ = lean_ctor_get(v___x_820_, 0);
lean_inc(v_fst_821_);
lean_dec(v___x_820_);
return v_fst_821_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxsValues___boxed(lean_object* v_00_u03b1_822_, lean_object* v_p_823_, lean_object* v_l_824_, lean_object* v_start_825_){
_start:
{
lean_object* v_res_826_; 
v_res_826_ = lp_batteries_List_findIdxsValues(v_00_u03b1_822_, v_p_823_, v_l_824_, v_start_825_);
lean_dec(v_start_825_);
return v_res_826_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxNth_go___redArg(lean_object* v_p_827_, lean_object* v_xs_828_, lean_object* v_n_829_, lean_object* v_s_830_){
_start:
{
if (lean_obj_tag(v_xs_828_) == 0)
{
lean_dec(v_n_829_);
lean_dec_ref(v_p_827_);
return v_s_830_;
}
else
{
lean_object* v_head_831_; lean_object* v_tail_832_; lean_object* v_zero_833_; uint8_t v_isZero_834_; 
v_head_831_ = lean_ctor_get(v_xs_828_, 0);
lean_inc(v_head_831_);
v_tail_832_ = lean_ctor_get(v_xs_828_, 1);
lean_inc(v_tail_832_);
lean_dec_ref_known(v_xs_828_, 2);
v_zero_833_ = lean_unsigned_to_nat(0u);
v_isZero_834_ = lean_nat_dec_eq(v_n_829_, v_zero_833_);
if (v_isZero_834_ == 1)
{
lean_object* v___x_835_; uint8_t v___x_836_; 
lean_dec(v_n_829_);
lean_inc_ref(v_p_827_);
v___x_835_ = lean_apply_1(v_p_827_, v_head_831_);
v___x_836_ = lean_unbox(v___x_835_);
if (v___x_836_ == 0)
{
lean_object* v___x_837_; lean_object* v___x_838_; 
v___x_837_ = lean_unsigned_to_nat(1u);
v___x_838_ = lean_nat_add(v_s_830_, v___x_837_);
lean_dec(v_s_830_);
v_xs_828_ = v_tail_832_;
v_n_829_ = v_zero_833_;
v_s_830_ = v___x_838_;
goto _start;
}
else
{
lean_dec(v_tail_832_);
lean_dec_ref(v_p_827_);
return v_s_830_;
}
}
else
{
lean_object* v_one_840_; lean_object* v_n_841_; lean_object* v___x_842_; uint8_t v___x_843_; 
v_one_840_ = lean_unsigned_to_nat(1u);
v_n_841_ = lean_nat_sub(v_n_829_, v_one_840_);
lean_dec(v_n_829_);
lean_inc_ref(v_p_827_);
v___x_842_ = lean_apply_1(v_p_827_, v_head_831_);
v___x_843_ = lean_unbox(v___x_842_);
if (v___x_843_ == 0)
{
lean_object* v___x_844_; lean_object* v___x_845_; 
v___x_844_ = lean_nat_add(v_n_841_, v_one_840_);
lean_dec(v_n_841_);
v___x_845_ = lean_nat_add(v_s_830_, v_one_840_);
lean_dec(v_s_830_);
v_xs_828_ = v_tail_832_;
v_n_829_ = v___x_844_;
v_s_830_ = v___x_845_;
goto _start;
}
else
{
lean_object* v___x_847_; 
v___x_847_ = lean_nat_add(v_s_830_, v_one_840_);
lean_dec(v_s_830_);
v_xs_828_ = v_tail_832_;
v_n_829_ = v_n_841_;
v_s_830_ = v___x_847_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxNth_go(lean_object* v_00_u03b1_849_, lean_object* v_p_850_, lean_object* v_xs_851_, lean_object* v_n_852_, lean_object* v_s_853_){
_start:
{
lean_object* v___x_854_; 
v___x_854_ = lp_batteries_List_findIdxNth_go___redArg(v_p_850_, v_xs_851_, v_n_852_, v_s_853_);
return v___x_854_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxNth___redArg(lean_object* v_p_855_, lean_object* v_xs_856_, lean_object* v_n_857_){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; 
v___x_858_ = lean_unsigned_to_nat(0u);
v___x_859_ = lp_batteries_List_findIdxNth_go___redArg(v_p_855_, v_xs_856_, v_n_857_, v___x_858_);
return v___x_859_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_findIdxNth(lean_object* v_00_u03b1_860_, lean_object* v_p_861_, lean_object* v_xs_862_, lean_object* v_n_863_){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; 
v___x_864_ = lean_unsigned_to_nat(0u);
v___x_865_ = lp_batteries_List_findIdxNth_go___redArg(v_p_861_, v_xs_862_, v_n_863_, v___x_864_);
return v___x_865_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf___redArg___lam__0(lean_object* v_inst_866_, lean_object* v_a_867_, lean_object* v_a_868_, lean_object* v_x_869_){
_start:
{
lean_object* v_fst_870_; lean_object* v_snd_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_886_; 
v_fst_870_ = lean_ctor_get(v_x_869_, 0);
v_snd_871_ = lean_ctor_get(v_x_869_, 1);
v_isSharedCheck_886_ = !lean_is_exclusive(v_x_869_);
if (v_isSharedCheck_886_ == 0)
{
v___x_873_ = v_x_869_;
v_isShared_874_ = v_isSharedCheck_886_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_snd_871_);
lean_inc(v_fst_870_);
lean_dec(v_x_869_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_886_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; uint8_t v___x_878_; 
v___x_875_ = lean_unsigned_to_nat(1u);
v___x_876_ = lean_nat_sub(v_snd_871_, v___x_875_);
lean_dec(v_snd_871_);
v___x_877_ = lean_apply_2(v_inst_866_, v_a_868_, v_a_867_);
v___x_878_ = lean_unbox(v___x_877_);
if (v___x_878_ == 0)
{
lean_object* v___x_880_; 
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 1, v___x_876_);
v___x_880_ = v___x_873_;
goto v_reusejp_879_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v_fst_870_);
lean_ctor_set(v_reuseFailAlloc_881_, 1, v___x_876_);
v___x_880_ = v_reuseFailAlloc_881_;
goto v_reusejp_879_;
}
v_reusejp_879_:
{
return v___x_880_;
}
}
else
{
lean_object* v___x_882_; lean_object* v___x_884_; 
lean_inc(v___x_876_);
v___x_882_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_882_, 0, v___x_876_);
lean_ctor_set(v___x_882_, 1, v_fst_870_);
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 1, v___x_876_);
lean_ctor_set(v___x_873_, 0, v___x_882_);
v___x_884_ = v___x_873_;
goto v_reusejp_883_;
}
else
{
lean_object* v_reuseFailAlloc_885_; 
v_reuseFailAlloc_885_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_885_, 0, v___x_882_);
lean_ctor_set(v_reuseFailAlloc_885_, 1, v___x_876_);
v___x_884_ = v_reuseFailAlloc_885_;
goto v_reusejp_883_;
}
v_reusejp_883_:
{
return v___x_884_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf___redArg(lean_object* v_inst_887_, lean_object* v_a_888_, lean_object* v_xs_889_, lean_object* v_start_890_){
_start:
{
lean_object* v___f_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v_fst_897_; 
v___f_891_ = lean_alloc_closure((void*)(lp_batteries_List_idxsOf___redArg___lam__0), 4, 2);
lean_closure_set(v___f_891_, 0, v_inst_887_);
lean_closure_set(v___f_891_, 1, v_a_888_);
v___x_892_ = lean_box(0);
v___x_893_ = l_List_lengthTR___redArg(v_xs_889_);
v___x_894_ = lean_nat_add(v_start_890_, v___x_893_);
lean_dec(v___x_893_);
v___x_895_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_895_, 0, v___x_892_);
lean_ctor_set(v___x_895_, 1, v___x_894_);
v___x_896_ = l_List_foldrTR___redArg(v___f_891_, v___x_895_, v_xs_889_);
v_fst_897_ = lean_ctor_get(v___x_896_, 0);
lean_inc(v_fst_897_);
lean_dec(v___x_896_);
return v_fst_897_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf___redArg___boxed(lean_object* v_inst_898_, lean_object* v_a_899_, lean_object* v_xs_900_, lean_object* v_start_901_){
_start:
{
lean_object* v_res_902_; 
v_res_902_ = lp_batteries_List_idxsOf___redArg(v_inst_898_, v_a_899_, v_xs_900_, v_start_901_);
lean_dec(v_start_901_);
return v_res_902_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf(lean_object* v_00_u03b1_903_, lean_object* v_inst_904_, lean_object* v_a_905_, lean_object* v_xs_906_, lean_object* v_start_907_){
_start:
{
lean_object* v___f_908_; lean_object* v___x_909_; lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v_fst_914_; 
v___f_908_ = lean_alloc_closure((void*)(lp_batteries_List_idxsOf___redArg___lam__0), 4, 2);
lean_closure_set(v___f_908_, 0, v_inst_904_);
lean_closure_set(v___f_908_, 1, v_a_905_);
v___x_909_ = lean_box(0);
v___x_910_ = l_List_lengthTR___redArg(v_xs_906_);
v___x_911_ = lean_nat_add(v_start_907_, v___x_910_);
lean_dec(v___x_910_);
v___x_912_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_912_, 0, v___x_909_);
lean_ctor_set(v___x_912_, 1, v___x_911_);
v___x_913_ = l_List_foldrTR___redArg(v___f_908_, v___x_912_, v_xs_906_);
v_fst_914_ = lean_ctor_get(v___x_913_, 0);
lean_inc(v_fst_914_);
lean_dec(v___x_913_);
return v_fst_914_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxsOf___boxed(lean_object* v_00_u03b1_915_, lean_object* v_inst_916_, lean_object* v_a_917_, lean_object* v_xs_918_, lean_object* v_start_919_){
_start:
{
lean_object* v_res_920_; 
v_res_920_ = lp_batteries_List_idxsOf(v_00_u03b1_915_, v_inst_916_, v_a_917_, v_xs_918_, v_start_919_);
lean_dec(v_start_919_);
return v_res_920_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_idxOfNth___redArg___lam__0(lean_object* v_inst_921_, lean_object* v_a_922_, lean_object* v_x_923_){
_start:
{
lean_object* v___x_924_; uint8_t v___x_925_; 
v___x_924_ = lean_apply_2(v_inst_921_, v_x_923_, v_a_922_);
v___x_925_ = lean_unbox(v___x_924_);
return v___x_925_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxOfNth___redArg___lam__0___boxed(lean_object* v_inst_926_, lean_object* v_a_927_, lean_object* v_x_928_){
_start:
{
uint8_t v_res_929_; lean_object* v_r_930_; 
v_res_929_ = lp_batteries_List_idxOfNth___redArg___lam__0(v_inst_926_, v_a_927_, v_x_928_);
v_r_930_ = lean_box(v_res_929_);
return v_r_930_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxOfNth___redArg(lean_object* v_inst_931_, lean_object* v_a_932_, lean_object* v_xs_933_, lean_object* v_n_934_){
_start:
{
lean_object* v___f_935_; lean_object* v___x_936_; lean_object* v___x_937_; 
v___f_935_ = lean_alloc_closure((void*)(lp_batteries_List_idxOfNth___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_935_, 0, v_inst_931_);
lean_closure_set(v___f_935_, 1, v_a_932_);
v___x_936_ = lean_unsigned_to_nat(0u);
v___x_937_ = lp_batteries_List_findIdxNth_go___redArg(v___f_935_, v_xs_933_, v_n_934_, v___x_936_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_idxOfNth(lean_object* v_00_u03b1_938_, lean_object* v_inst_939_, lean_object* v_a_940_, lean_object* v_xs_941_, lean_object* v_n_942_){
_start:
{
lean_object* v___x_943_; 
v___x_943_ = lp_batteries_List_idxOfNth___redArg(v_inst_939_, v_a_940_, v_xs_941_, v_n_942_);
return v___x_943_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore_go___redArg(lean_object* v_p_944_, lean_object* v_xs_945_, lean_object* v_i_946_, lean_object* v_s_947_){
_start:
{
if (lean_obj_tag(v_xs_945_) == 0)
{
lean_dec(v_i_946_);
lean_dec_ref(v_p_944_);
return v_s_947_;
}
else
{
lean_object* v_head_948_; lean_object* v_tail_949_; lean_object* v_zero_950_; uint8_t v_isZero_951_; 
v_head_948_ = lean_ctor_get(v_xs_945_, 0);
lean_inc(v_head_948_);
v_tail_949_ = lean_ctor_get(v_xs_945_, 1);
lean_inc(v_tail_949_);
lean_dec_ref_known(v_xs_945_, 2);
v_zero_950_ = lean_unsigned_to_nat(0u);
v_isZero_951_ = lean_nat_dec_eq(v_i_946_, v_zero_950_);
if (v_isZero_951_ == 1)
{
lean_dec(v_tail_949_);
lean_dec(v_head_948_);
lean_dec(v_i_946_);
lean_dec_ref(v_p_944_);
return v_s_947_;
}
else
{
lean_object* v_one_952_; lean_object* v_n_953_; lean_object* v___x_954_; uint8_t v___x_955_; 
v_one_952_ = lean_unsigned_to_nat(1u);
v_n_953_ = lean_nat_sub(v_i_946_, v_one_952_);
lean_dec(v_i_946_);
lean_inc_ref(v_p_944_);
v___x_954_ = lean_apply_1(v_p_944_, v_head_948_);
v___x_955_ = lean_unbox(v___x_954_);
if (v___x_955_ == 0)
{
v_xs_945_ = v_tail_949_;
v_i_946_ = v_n_953_;
goto _start;
}
else
{
lean_object* v___x_957_; 
v___x_957_ = lean_nat_add(v_s_947_, v_one_952_);
lean_dec(v_s_947_);
v_xs_945_ = v_tail_949_;
v_i_946_ = v_n_953_;
v_s_947_ = v___x_957_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore_go(lean_object* v_00_u03b1_959_, lean_object* v_p_960_, lean_object* v_xs_961_, lean_object* v_i_962_, lean_object* v_s_963_){
_start:
{
lean_object* v___x_964_; 
v___x_964_ = lp_batteries_List_countPBefore_go___redArg(v_p_960_, v_xs_961_, v_i_962_, v_s_963_);
return v___x_964_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore_go___at___00List_countPBefore_spec__0___redArg(lean_object* v_p_965_, lean_object* v_xs_966_, lean_object* v_i_967_, lean_object* v_s_968_){
_start:
{
if (lean_obj_tag(v_xs_966_) == 0)
{
lean_dec(v_i_967_);
lean_dec_ref(v_p_965_);
return v_s_968_;
}
else
{
lean_object* v_head_969_; lean_object* v_tail_970_; lean_object* v_zero_971_; uint8_t v_isZero_972_; 
v_head_969_ = lean_ctor_get(v_xs_966_, 0);
lean_inc(v_head_969_);
v_tail_970_ = lean_ctor_get(v_xs_966_, 1);
lean_inc(v_tail_970_);
lean_dec_ref_known(v_xs_966_, 2);
v_zero_971_ = lean_unsigned_to_nat(0u);
v_isZero_972_ = lean_nat_dec_eq(v_i_967_, v_zero_971_);
if (v_isZero_972_ == 1)
{
lean_dec(v_tail_970_);
lean_dec(v_head_969_);
lean_dec(v_i_967_);
lean_dec_ref(v_p_965_);
return v_s_968_;
}
else
{
lean_object* v_one_973_; lean_object* v_n_974_; lean_object* v___x_975_; uint8_t v___x_976_; 
v_one_973_ = lean_unsigned_to_nat(1u);
v_n_974_ = lean_nat_sub(v_i_967_, v_one_973_);
lean_dec(v_i_967_);
lean_inc_ref(v_p_965_);
v___x_975_ = lean_apply_1(v_p_965_, v_head_969_);
v___x_976_ = lean_unbox(v___x_975_);
if (v___x_976_ == 0)
{
v_xs_966_ = v_tail_970_;
v_i_967_ = v_n_974_;
goto _start;
}
else
{
lean_object* v___x_978_; 
v___x_978_ = lean_nat_add(v_s_968_, v_one_973_);
lean_dec(v_s_968_);
v_xs_966_ = v_tail_970_;
v_i_967_ = v_n_974_;
v_s_968_ = v___x_978_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore___redArg(lean_object* v_p_980_, lean_object* v_xs_981_, lean_object* v_i_982_){
_start:
{
lean_object* v___x_983_; lean_object* v___x_984_; 
v___x_983_ = lean_unsigned_to_nat(0u);
v___x_984_ = lp_batteries_List_countPBefore_go___at___00List_countPBefore_spec__0___redArg(v_p_980_, v_xs_981_, v_i_982_, v___x_983_);
return v___x_984_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore(lean_object* v_00_u03b1_985_, lean_object* v_p_986_, lean_object* v_xs_987_, lean_object* v_i_988_){
_start:
{
lean_object* v___x_989_; 
v___x_989_ = lp_batteries_List_countPBefore___redArg(v_p_986_, v_xs_987_, v_i_988_);
return v___x_989_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_countPBefore_go___at___00List_countPBefore_spec__0(lean_object* v_00_u03b1_990_, lean_object* v_p_991_, lean_object* v_xs_992_, lean_object* v_i_993_, lean_object* v_s_994_){
_start:
{
lean_object* v___x_995_; 
v___x_995_ = lp_batteries_List_countPBefore_go___at___00List_countPBefore_spec__0___redArg(v_p_991_, v_xs_992_, v_i_993_, v_s_994_);
return v___x_995_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_countBefore___redArg(lean_object* v_inst_996_, lean_object* v_a_997_, lean_object* v_xs_998_, lean_object* v_i_999_){
_start:
{
lean_object* v___f_1000_; lean_object* v___x_1001_; 
v___f_1000_ = lean_alloc_closure((void*)(lp_batteries_List_idxOfNth___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1000_, 0, v_inst_996_);
lean_closure_set(v___f_1000_, 1, v_a_997_);
v___x_1001_ = lp_batteries_List_countPBefore___redArg(v___f_1000_, v_xs_998_, v_i_999_);
return v___x_1001_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_countBefore(lean_object* v_00_u03b1_1002_, lean_object* v_inst_1003_, lean_object* v_a_1004_, lean_object* v_xs_1005_, lean_object* v_i_1006_){
_start:
{
lean_object* v___x_1007_; 
v___x_1007_ = lp_batteries_List_countBefore___redArg(v_inst_1003_, v_a_1004_, v_xs_1005_, v_i_1006_);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_lookmap_go___redArg(lean_object* v_f_1008_, lean_object* v_a_1009_, lean_object* v_a_1010_){
_start:
{
if (lean_obj_tag(v_a_1009_) == 0)
{
lean_object* v___x_1011_; 
lean_dec_ref(v_f_1008_);
v___x_1011_ = lean_array_to_list(v_a_1010_);
return v___x_1011_;
}
else
{
lean_object* v_head_1012_; lean_object* v_tail_1013_; lean_object* v___x_1015_; uint8_t v_isShared_1016_; uint8_t v_isSharedCheck_1032_; 
v_head_1012_ = lean_ctor_get(v_a_1009_, 0);
v_tail_1013_ = lean_ctor_get(v_a_1009_, 1);
v_isSharedCheck_1032_ = !lean_is_exclusive(v_a_1009_);
if (v_isSharedCheck_1032_ == 0)
{
v___x_1015_ = v_a_1009_;
v_isShared_1016_ = v_isSharedCheck_1032_;
goto v_resetjp_1014_;
}
else
{
lean_inc(v_tail_1013_);
lean_inc(v_head_1012_);
lean_dec(v_a_1009_);
v___x_1015_ = lean_box(0);
v_isShared_1016_ = v_isSharedCheck_1032_;
goto v_resetjp_1014_;
}
v_resetjp_1014_:
{
lean_object* v___x_1017_; 
lean_inc_ref(v_f_1008_);
lean_inc(v_head_1012_);
v___x_1017_ = lean_apply_1(v_f_1008_, v_head_1012_);
if (lean_obj_tag(v___x_1017_) == 0)
{
lean_object* v___x_1018_; 
lean_del_object(v___x_1015_);
v___x_1018_ = lean_array_push(v_a_1010_, v_head_1012_);
v_a_1009_ = v_tail_1013_;
v_a_1010_ = v___x_1018_;
goto _start;
}
else
{
lean_object* v_val_1020_; lean_object* v___x_1022_; 
lean_dec(v_head_1012_);
lean_dec_ref(v_f_1008_);
v_val_1020_ = lean_ctor_get(v___x_1017_, 0);
lean_inc(v_val_1020_);
lean_dec_ref_known(v___x_1017_, 1);
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 0, v_val_1020_);
v___x_1022_ = v___x_1015_;
goto v_reusejp_1021_;
}
else
{
lean_object* v_reuseFailAlloc_1031_; 
v_reuseFailAlloc_1031_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1031_, 0, v_val_1020_);
lean_ctor_set(v_reuseFailAlloc_1031_, 1, v_tail_1013_);
v___x_1022_ = v_reuseFailAlloc_1031_;
goto v_reusejp_1021_;
}
v_reusejp_1021_:
{
lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; uint8_t v___x_1026_; 
v___x_1023_ = lean_array_get_size(v_a_1010_);
v___x_1024_ = lean_unsigned_to_nat(0u);
v___x_1025_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__9));
v___x_1026_ = lean_nat_dec_lt(v___x_1024_, v___x_1023_);
if (v___x_1026_ == 0)
{
lean_dec_ref(v_a_1010_);
return v___x_1022_;
}
else
{
lean_object* v___f_1027_; size_t v___x_1028_; size_t v___x_1029_; lean_object* v___x_1030_; 
v___f_1027_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__10));
v___x_1028_ = lean_usize_of_nat(v___x_1023_);
v___x_1029_ = ((size_t)0ULL);
v___x_1030_ = l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1025_, v___f_1027_, v_a_1010_, v___x_1028_, v___x_1029_, v___x_1022_);
return v___x_1030_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_lookmap_go(lean_object* v_00_u03b1_1033_, lean_object* v_f_1034_, lean_object* v_a_1035_, lean_object* v_a_1036_){
_start:
{
lean_object* v___x_1037_; 
v___x_1037_ = lp_batteries_List_lookmap_go___redArg(v_f_1034_, v_a_1035_, v_a_1036_);
return v___x_1037_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_lookmap___redArg(lean_object* v_f_1038_, lean_object* v_l_1039_){
_start:
{
lean_object* v___x_1040_; lean_object* v___x_1041_; 
v___x_1040_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_1041_ = lp_batteries_List_lookmap_go___redArg(v_f_1038_, v_l_1039_, v___x_1040_);
return v___x_1041_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_lookmap(lean_object* v_00_u03b1_1042_, lean_object* v_f_1043_, lean_object* v_l_1044_){
_start:
{
lean_object* v___x_1045_; lean_object* v___x_1046_; 
v___x_1045_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_1046_ = lp_batteries_List_lookmap_go___redArg(v_f_1043_, v_l_1044_, v___x_1045_);
return v___x_1046_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_inits_spec__0___redArg(lean_object* v_head_1047_, lean_object* v_a_1048_, lean_object* v_a_1049_){
_start:
{
if (lean_obj_tag(v_a_1048_) == 0)
{
lean_object* v___x_1050_; 
lean_dec(v_head_1047_);
v___x_1050_ = l_List_reverse___redArg(v_a_1049_);
return v___x_1050_;
}
else
{
lean_object* v_head_1051_; lean_object* v_tail_1052_; lean_object* v___x_1054_; uint8_t v_isShared_1055_; uint8_t v_isSharedCheck_1061_; 
v_head_1051_ = lean_ctor_get(v_a_1048_, 0);
v_tail_1052_ = lean_ctor_get(v_a_1048_, 1);
v_isSharedCheck_1061_ = !lean_is_exclusive(v_a_1048_);
if (v_isSharedCheck_1061_ == 0)
{
v___x_1054_ = v_a_1048_;
v_isShared_1055_ = v_isSharedCheck_1061_;
goto v_resetjp_1053_;
}
else
{
lean_inc(v_tail_1052_);
lean_inc(v_head_1051_);
lean_dec(v_a_1048_);
v___x_1054_ = lean_box(0);
v_isShared_1055_ = v_isSharedCheck_1061_;
goto v_resetjp_1053_;
}
v_resetjp_1053_:
{
lean_object* v___x_1057_; 
lean_inc(v_head_1047_);
if (v_isShared_1055_ == 0)
{
lean_ctor_set(v___x_1054_, 1, v_head_1051_);
lean_ctor_set(v___x_1054_, 0, v_head_1047_);
v___x_1057_ = v___x_1054_;
goto v_reusejp_1056_;
}
else
{
lean_object* v_reuseFailAlloc_1060_; 
v_reuseFailAlloc_1060_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1060_, 0, v_head_1047_);
lean_ctor_set(v_reuseFailAlloc_1060_, 1, v_head_1051_);
v___x_1057_ = v_reuseFailAlloc_1060_;
goto v_reusejp_1056_;
}
v_reusejp_1056_:
{
lean_object* v___x_1058_; 
v___x_1058_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1058_, 0, v___x_1057_);
lean_ctor_set(v___x_1058_, 1, v_a_1049_);
v_a_1048_ = v_tail_1052_;
v_a_1049_ = v___x_1058_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_inits___redArg(lean_object* v_x_1062_){
_start:
{
if (lean_obj_tag(v_x_1062_) == 0)
{
lean_object* v___x_1063_; lean_object* v___x_1064_; 
v___x_1063_ = lean_box(0);
v___x_1064_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1064_, 0, v_x_1062_);
lean_ctor_set(v___x_1064_, 1, v___x_1063_);
return v___x_1064_;
}
else
{
lean_object* v_head_1065_; lean_object* v_tail_1066_; lean_object* v___x_1068_; uint8_t v_isShared_1069_; uint8_t v_isSharedCheck_1076_; 
v_head_1065_ = lean_ctor_get(v_x_1062_, 0);
v_tail_1066_ = lean_ctor_get(v_x_1062_, 1);
v_isSharedCheck_1076_ = !lean_is_exclusive(v_x_1062_);
if (v_isSharedCheck_1076_ == 0)
{
v___x_1068_ = v_x_1062_;
v_isShared_1069_ = v_isSharedCheck_1076_;
goto v_resetjp_1067_;
}
else
{
lean_inc(v_tail_1066_);
lean_inc(v_head_1065_);
lean_dec(v_x_1062_);
v___x_1068_ = lean_box(0);
v_isShared_1069_ = v_isSharedCheck_1076_;
goto v_resetjp_1067_;
}
v_resetjp_1067_:
{
lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1074_; 
v___x_1070_ = lean_box(0);
v___x_1071_ = lp_batteries_List_inits___redArg(v_tail_1066_);
v___x_1072_ = lp_batteries_List_mapTR_loop___at___00List_inits_spec__0___redArg(v_head_1065_, v___x_1071_, v___x_1070_);
if (v_isShared_1069_ == 0)
{
lean_ctor_set(v___x_1068_, 1, v___x_1072_);
lean_ctor_set(v___x_1068_, 0, v___x_1070_);
v___x_1074_ = v___x_1068_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v___x_1070_);
lean_ctor_set(v_reuseFailAlloc_1075_, 1, v___x_1072_);
v___x_1074_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
return v___x_1074_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_inits(lean_object* v_00_u03b1_1077_, lean_object* v_x_1078_){
_start:
{
lean_object* v___x_1079_; 
v___x_1079_ = lp_batteries_List_inits___redArg(v_x_1078_);
return v___x_1079_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_inits_spec__0(lean_object* v_00_u03b1_1080_, lean_object* v_head_1081_, lean_object* v_a_1082_, lean_object* v_a_1083_){
_start:
{
lean_object* v___x_1084_; 
v___x_1084_ = lp_batteries_List_mapTR_loop___at___00List_inits_spec__0___redArg(v_head_1081_, v_a_1082_, v_a_1083_);
return v___x_1084_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___redArg(lean_object* v_a_1085_, size_t v_sz_1086_, size_t v_i_1087_, lean_object* v_bs_1088_){
_start:
{
uint8_t v___x_1089_; 
v___x_1089_ = lean_usize_dec_lt(v_i_1087_, v_sz_1086_);
if (v___x_1089_ == 0)
{
lean_dec(v_a_1085_);
return v_bs_1088_;
}
else
{
lean_object* v_v_1090_; lean_object* v___x_1091_; lean_object* v_bs_x27_1092_; lean_object* v___x_1093_; size_t v___x_1094_; size_t v___x_1095_; lean_object* v___x_1096_; 
v_v_1090_ = lean_array_uget(v_bs_1088_, v_i_1087_);
v___x_1091_ = lean_unsigned_to_nat(0u);
v_bs_x27_1092_ = lean_array_uset(v_bs_1088_, v_i_1087_, v___x_1091_);
lean_inc(v_a_1085_);
v___x_1093_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1093_, 0, v_a_1085_);
lean_ctor_set(v___x_1093_, 1, v_v_1090_);
v___x_1094_ = ((size_t)1ULL);
v___x_1095_ = lean_usize_add(v_i_1087_, v___x_1094_);
v___x_1096_ = lean_array_uset(v_bs_x27_1092_, v_i_1087_, v___x_1093_);
v_i_1087_ = v___x_1095_;
v_bs_1088_ = v___x_1096_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___redArg___boxed(lean_object* v_a_1098_, lean_object* v_sz_1099_, lean_object* v_i_1100_, lean_object* v_bs_1101_){
_start:
{
size_t v_sz_boxed_1102_; size_t v_i_boxed_1103_; lean_object* v_res_1104_; 
v_sz_boxed_1102_ = lean_unbox_usize(v_sz_1099_);
lean_dec(v_sz_1099_);
v_i_boxed_1103_ = lean_unbox_usize(v_i_1100_);
lean_dec(v_i_1100_);
v_res_1104_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___redArg(v_a_1098_, v_sz_boxed_1102_, v_i_boxed_1103_, v_bs_1101_);
return v_res_1104_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___redArg(lean_object* v_as_1105_, size_t v_i_1106_, size_t v_stop_1107_, lean_object* v_b_1108_){
_start:
{
uint8_t v___x_1109_; 
v___x_1109_ = lean_usize_dec_eq(v_i_1106_, v_stop_1107_);
if (v___x_1109_ == 0)
{
size_t v___x_1110_; size_t v___x_1111_; lean_object* v___x_1112_; size_t v_sz_1113_; size_t v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; 
v___x_1110_ = ((size_t)1ULL);
v___x_1111_ = lean_usize_sub(v_i_1106_, v___x_1110_);
v___x_1112_ = lean_array_uget_borrowed(v_as_1105_, v___x_1111_);
v_sz_1113_ = lean_array_size(v_b_1108_);
v___x_1114_ = ((size_t)0ULL);
lean_inc(v___x_1112_);
v___x_1115_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___redArg(v___x_1112_, v_sz_1113_, v___x_1114_, v_b_1108_);
v___x_1116_ = lean_box(0);
v___x_1117_ = lean_array_push(v___x_1115_, v___x_1116_);
v_i_1106_ = v___x_1111_;
v_b_1108_ = v___x_1117_;
goto _start;
}
else
{
return v_b_1108_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___redArg___boxed(lean_object* v_as_1119_, lean_object* v_i_1120_, lean_object* v_stop_1121_, lean_object* v_b_1122_){
_start:
{
size_t v_i_boxed_1123_; size_t v_stop_boxed_1124_; lean_object* v_res_1125_; 
v_i_boxed_1123_ = lean_unbox_usize(v_i_1120_);
lean_dec(v_i_1120_);
v_stop_boxed_1124_ = lean_unbox_usize(v_stop_1121_);
lean_dec(v_stop_1121_);
v_res_1125_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___redArg(v_as_1119_, v_i_boxed_1123_, v_stop_boxed_1124_, v_b_1122_);
lean_dec_ref(v_as_1119_);
return v_res_1125_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_initsTR_spec__1___redArg(lean_object* v_init_1126_, lean_object* v_l_1127_){
_start:
{
lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; uint8_t v___x_1131_; 
v___x_1128_ = lean_array_mk(v_l_1127_);
v___x_1129_ = lean_array_get_size(v___x_1128_);
v___x_1130_ = lean_unsigned_to_nat(0u);
v___x_1131_ = lean_nat_dec_lt(v___x_1130_, v___x_1129_);
if (v___x_1131_ == 0)
{
lean_dec_ref(v___x_1128_);
return v_init_1126_;
}
else
{
size_t v___x_1132_; size_t v___x_1133_; lean_object* v___x_1134_; 
v___x_1132_ = lean_usize_of_nat(v___x_1129_);
v___x_1133_ = ((size_t)0ULL);
v___x_1134_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___redArg(v___x_1128_, v___x_1132_, v___x_1133_, v_init_1126_);
lean_dec_ref(v___x_1128_);
return v___x_1134_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___redArg(lean_object* v_as_1135_, size_t v_i_1136_, size_t v_stop_1137_, lean_object* v_b_1138_){
_start:
{
uint8_t v___x_1139_; 
v___x_1139_ = lean_usize_dec_eq(v_i_1136_, v_stop_1137_);
if (v___x_1139_ == 0)
{
lean_object* v___x_1140_; lean_object* v___x_1141_; size_t v___x_1142_; size_t v___x_1143_; 
v___x_1140_ = lean_array_uget_borrowed(v_as_1135_, v_i_1136_);
lean_inc(v___x_1140_);
v___x_1141_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1141_, 0, v___x_1140_);
lean_ctor_set(v___x_1141_, 1, v_b_1138_);
v___x_1142_ = ((size_t)1ULL);
v___x_1143_ = lean_usize_add(v_i_1136_, v___x_1142_);
v_i_1136_ = v___x_1143_;
v_b_1138_ = v___x_1141_;
goto _start;
}
else
{
return v_b_1138_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___redArg___boxed(lean_object* v_as_1145_, lean_object* v_i_1146_, lean_object* v_stop_1147_, lean_object* v_b_1148_){
_start:
{
size_t v_i_boxed_1149_; size_t v_stop_boxed_1150_; lean_object* v_res_1151_; 
v_i_boxed_1149_ = lean_unbox_usize(v_i_1146_);
lean_dec(v_i_1146_);
v_stop_boxed_1150_ = lean_unbox_usize(v_stop_1147_);
lean_dec(v_stop_1147_);
v_res_1151_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___redArg(v_as_1145_, v_i_boxed_1149_, v_stop_boxed_1150_, v_b_1148_);
lean_dec_ref(v_as_1145_);
return v_res_1151_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_initsTR___redArg(lean_object* v_l_1156_){
_start:
{
lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; uint8_t v___x_1162_; 
v___x_1157_ = lean_box(0);
v___x_1158_ = ((lean_object*)(lp_batteries_List_initsTR___redArg___closed__0));
v___x_1159_ = lp_batteries_List_foldrTR___at___00List_initsTR_spec__1___redArg(v___x_1158_, v_l_1156_);
v___x_1160_ = lean_unsigned_to_nat(0u);
v___x_1161_ = lean_array_get_size(v___x_1159_);
v___x_1162_ = lean_nat_dec_lt(v___x_1160_, v___x_1161_);
if (v___x_1162_ == 0)
{
lean_dec_ref(v___x_1159_);
return v___x_1157_;
}
else
{
uint8_t v___x_1163_; 
v___x_1163_ = lean_nat_dec_le(v___x_1161_, v___x_1161_);
if (v___x_1163_ == 0)
{
if (v___x_1162_ == 0)
{
lean_dec_ref(v___x_1159_);
return v___x_1157_;
}
else
{
size_t v___x_1164_; size_t v___x_1165_; lean_object* v___x_1166_; 
v___x_1164_ = ((size_t)0ULL);
v___x_1165_ = lean_usize_of_nat(v___x_1161_);
v___x_1166_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___redArg(v___x_1159_, v___x_1164_, v___x_1165_, v___x_1157_);
lean_dec_ref(v___x_1159_);
return v___x_1166_;
}
}
else
{
size_t v___x_1167_; size_t v___x_1168_; lean_object* v___x_1169_; 
v___x_1167_ = ((size_t)0ULL);
v___x_1168_ = lean_usize_of_nat(v___x_1161_);
v___x_1169_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___redArg(v___x_1159_, v___x_1167_, v___x_1168_, v___x_1157_);
lean_dec_ref(v___x_1159_);
return v___x_1169_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_initsTR(lean_object* v_00_u03b1_1170_, lean_object* v_l_1171_){
_start:
{
lean_object* v___x_1172_; 
v___x_1172_ = lp_batteries_List_initsTR___redArg(v_l_1171_);
return v___x_1172_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0(lean_object* v_00_u03b1_1173_, lean_object* v_a_1174_, size_t v_sz_1175_, size_t v_i_1176_, lean_object* v_bs_1177_){
_start:
{
lean_object* v___x_1178_; 
v___x_1178_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___redArg(v_a_1174_, v_sz_1175_, v_i_1176_, v_bs_1177_);
return v___x_1178_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0___boxed(lean_object* v_00_u03b1_1179_, lean_object* v_a_1180_, lean_object* v_sz_1181_, lean_object* v_i_1182_, lean_object* v_bs_1183_){
_start:
{
size_t v_sz_boxed_1184_; size_t v_i_boxed_1185_; lean_object* v_res_1186_; 
v_sz_boxed_1184_ = lean_unbox_usize(v_sz_1181_);
lean_dec(v_sz_1181_);
v_i_boxed_1185_ = lean_unbox_usize(v_i_1182_);
lean_dec(v_i_1182_);
v_res_1186_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_initsTR_spec__0(v_00_u03b1_1179_, v_a_1180_, v_sz_boxed_1184_, v_i_boxed_1185_, v_bs_1183_);
return v_res_1186_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_initsTR_spec__1(lean_object* v_00_u03b1_1187_, lean_object* v_init_1188_, lean_object* v_l_1189_){
_start:
{
lean_object* v___x_1190_; 
v___x_1190_ = lp_batteries_List_foldrTR___at___00List_initsTR_spec__1___redArg(v_init_1188_, v_l_1189_);
return v___x_1190_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2(lean_object* v_00_u03b1_1191_, lean_object* v_as_1192_, size_t v_i_1193_, size_t v_stop_1194_, lean_object* v_b_1195_){
_start:
{
lean_object* v___x_1196_; 
v___x_1196_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___redArg(v_as_1192_, v_i_1193_, v_stop_1194_, v_b_1195_);
return v___x_1196_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2___boxed(lean_object* v_00_u03b1_1197_, lean_object* v_as_1198_, lean_object* v_i_1199_, lean_object* v_stop_1200_, lean_object* v_b_1201_){
_start:
{
size_t v_i_boxed_1202_; size_t v_stop_boxed_1203_; lean_object* v_res_1204_; 
v_i_boxed_1202_ = lean_unbox_usize(v_i_1199_);
lean_dec(v_i_1199_);
v_stop_boxed_1203_ = lean_unbox_usize(v_stop_1200_);
lean_dec(v_stop_1200_);
v_res_1204_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_initsTR_spec__2(v_00_u03b1_1197_, v_as_1198_, v_i_boxed_1202_, v_stop_boxed_1203_, v_b_1201_);
lean_dec_ref(v_as_1198_);
return v_res_1204_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1(lean_object* v_00_u03b1_1205_, lean_object* v_as_1206_, size_t v_i_1207_, size_t v_stop_1208_, lean_object* v_b_1209_){
_start:
{
lean_object* v___x_1210_; 
v___x_1210_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___redArg(v_as_1206_, v_i_1207_, v_stop_1208_, v_b_1209_);
return v___x_1210_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1___boxed(lean_object* v_00_u03b1_1211_, lean_object* v_as_1212_, lean_object* v_i_1213_, lean_object* v_stop_1214_, lean_object* v_b_1215_){
_start:
{
size_t v_i_boxed_1216_; size_t v_stop_boxed_1217_; lean_object* v_res_1218_; 
v_i_boxed_1216_ = lean_unbox_usize(v_i_1213_);
lean_dec(v_i_1213_);
v_stop_boxed_1217_ = lean_unbox_usize(v_stop_1214_);
lean_dec(v_stop_1214_);
v_res_1218_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_initsTR_spec__1_spec__1(v_00_u03b1_1211_, v_as_1212_, v_i_boxed_1216_, v_stop_boxed_1217_, v_b_1215_);
lean_dec_ref(v_as_1212_);
return v_res_1218_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_tails___redArg(lean_object* v_x_1219_){
_start:
{
if (lean_obj_tag(v_x_1219_) == 0)
{
lean_object* v___x_1220_; lean_object* v___x_1221_; 
v___x_1220_ = lean_box(0);
v___x_1221_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1221_, 0, v_x_1219_);
lean_ctor_set(v___x_1221_, 1, v___x_1220_);
return v___x_1221_;
}
else
{
lean_object* v_tail_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; 
v_tail_1222_ = lean_ctor_get(v_x_1219_, 1);
lean_inc(v_tail_1222_);
v___x_1223_ = lp_batteries_List_tails___redArg(v_tail_1222_);
v___x_1224_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1224_, 0, v_x_1219_);
lean_ctor_set(v___x_1224_, 1, v___x_1223_);
return v___x_1224_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_tails(lean_object* v_00_u03b1_1225_, lean_object* v_x_1226_){
_start:
{
lean_object* v___x_1227_; 
v___x_1227_ = lp_batteries_List_tails___redArg(v_x_1226_);
return v___x_1227_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___redArg(lean_object* v_as_1228_, size_t v_i_1229_, size_t v_stop_1230_, lean_object* v_b_1231_){
_start:
{
uint8_t v___x_1232_; 
v___x_1232_ = lean_usize_dec_eq(v_i_1229_, v_stop_1230_);
if (v___x_1232_ == 0)
{
size_t v___x_1233_; size_t v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; 
v___x_1233_ = ((size_t)1ULL);
v___x_1234_ = lean_usize_sub(v_i_1229_, v___x_1233_);
v___x_1235_ = lean_array_uget_borrowed(v_as_1228_, v___x_1234_);
lean_inc(v___x_1235_);
v___x_1236_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1236_, 0, v___x_1235_);
lean_ctor_set(v___x_1236_, 1, v_b_1231_);
v_i_1229_ = v___x_1234_;
v_b_1231_ = v___x_1236_;
goto _start;
}
else
{
return v_b_1231_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___redArg___boxed(lean_object* v_as_1238_, lean_object* v_i_1239_, lean_object* v_stop_1240_, lean_object* v_b_1241_){
_start:
{
size_t v_i_boxed_1242_; size_t v_stop_boxed_1243_; lean_object* v_res_1244_; 
v_i_boxed_1242_ = lean_unbox_usize(v_i_1239_);
lean_dec(v_i_1239_);
v_stop_boxed_1243_ = lean_unbox_usize(v_stop_1240_);
lean_dec(v_stop_1240_);
v_res_1244_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___redArg(v_as_1238_, v_i_boxed_1242_, v_stop_boxed_1243_, v_b_1241_);
lean_dec_ref(v_as_1238_);
return v_res_1244_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_tailsTR_go___redArg(lean_object* v_l_1245_, lean_object* v_acc_1246_){
_start:
{
if (lean_obj_tag(v_l_1245_) == 0)
{
lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; uint8_t v___x_1251_; 
v___x_1247_ = lean_box(0);
v___x_1248_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1248_, 0, v_l_1245_);
lean_ctor_set(v___x_1248_, 1, v___x_1247_);
v___x_1249_ = lean_array_get_size(v_acc_1246_);
v___x_1250_ = lean_unsigned_to_nat(0u);
v___x_1251_ = lean_nat_dec_lt(v___x_1250_, v___x_1249_);
if (v___x_1251_ == 0)
{
lean_dec_ref(v_acc_1246_);
return v___x_1248_;
}
else
{
size_t v___x_1252_; size_t v___x_1253_; lean_object* v___x_1254_; 
v___x_1252_ = lean_usize_of_nat(v___x_1249_);
v___x_1253_ = ((size_t)0ULL);
v___x_1254_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___redArg(v_acc_1246_, v___x_1252_, v___x_1253_, v___x_1248_);
lean_dec_ref(v_acc_1246_);
return v___x_1254_;
}
}
else
{
lean_object* v_tail_1255_; lean_object* v___x_1256_; 
v_tail_1255_ = lean_ctor_get(v_l_1245_, 1);
lean_inc(v_tail_1255_);
v___x_1256_ = lean_array_push(v_acc_1246_, v_l_1245_);
v_l_1245_ = v_tail_1255_;
v_acc_1246_ = v___x_1256_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_tailsTR_go(lean_object* v_00_u03b1_1258_, lean_object* v_l_1259_, lean_object* v_acc_1260_){
_start:
{
lean_object* v___x_1261_; 
v___x_1261_ = lp_batteries_List_tailsTR_go___redArg(v_l_1259_, v_acc_1260_);
return v___x_1261_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0(lean_object* v_00_u03b1_1262_, lean_object* v_as_1263_, size_t v_i_1264_, size_t v_stop_1265_, lean_object* v_b_1266_){
_start:
{
lean_object* v___x_1267_; 
v___x_1267_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___redArg(v_as_1263_, v_i_1264_, v_stop_1265_, v_b_1266_);
return v___x_1267_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0___boxed(lean_object* v_00_u03b1_1268_, lean_object* v_as_1269_, lean_object* v_i_1270_, lean_object* v_stop_1271_, lean_object* v_b_1272_){
_start:
{
size_t v_i_boxed_1273_; size_t v_stop_boxed_1274_; lean_object* v_res_1275_; 
v_i_boxed_1273_ = lean_unbox_usize(v_i_1270_);
lean_dec(v_i_1270_);
v_stop_boxed_1274_ = lean_unbox_usize(v_stop_1271_);
lean_dec(v_stop_1271_);
v_res_1275_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_tailsTR_go_spec__0(v_00_u03b1_1268_, v_as_1269_, v_i_boxed_1273_, v_stop_boxed_1274_, v_b_1272_);
lean_dec_ref(v_as_1269_);
return v_res_1275_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_tailsTR___redArg(lean_object* v_l_1278_){
_start:
{
lean_object* v___x_1279_; lean_object* v___x_1280_; 
v___x_1279_ = ((lean_object*)(lp_batteries_List_tailsTR___redArg___closed__0));
v___x_1280_ = lp_batteries_List_tailsTR_go___redArg(v_l_1278_, v___x_1279_);
return v___x_1280_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_tailsTR(lean_object* v_00_u03b1_1281_, lean_object* v_l_1282_){
_start:
{
lean_object* v___x_1283_; 
v___x_1283_ = lp_batteries_List_tailsTR___redArg(v_l_1282_);
return v___x_1283_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___redArg(lean_object* v_a_1284_, lean_object* v_as_1285_, size_t v_i_1286_, size_t v_stop_1287_, lean_object* v_b_1288_){
_start:
{
uint8_t v___x_1289_; 
v___x_1289_ = lean_usize_dec_eq(v_i_1286_, v_stop_1287_);
if (v___x_1289_ == 0)
{
lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; size_t v___x_1293_; size_t v___x_1294_; 
v___x_1290_ = lean_array_uget_borrowed(v_as_1285_, v_i_1286_);
lean_inc(v___x_1290_);
lean_inc(v_a_1284_);
v___x_1291_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1291_, 0, v_a_1284_);
lean_ctor_set(v___x_1291_, 1, v___x_1290_);
v___x_1292_ = lean_array_push(v_b_1288_, v___x_1291_);
v___x_1293_ = ((size_t)1ULL);
v___x_1294_ = lean_usize_add(v_i_1286_, v___x_1293_);
v_i_1286_ = v___x_1294_;
v_b_1288_ = v___x_1292_;
goto _start;
}
else
{
lean_dec(v_a_1284_);
return v_b_1288_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___redArg___boxed(lean_object* v_a_1296_, lean_object* v_as_1297_, lean_object* v_i_1298_, lean_object* v_stop_1299_, lean_object* v_b_1300_){
_start:
{
size_t v_i_boxed_1301_; size_t v_stop_boxed_1302_; lean_object* v_res_1303_; 
v_i_boxed_1301_ = lean_unbox_usize(v_i_1298_);
lean_dec(v_i_1298_);
v_stop_boxed_1302_ = lean_unbox_usize(v_stop_1299_);
lean_dec(v_stop_1299_);
v_res_1303_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___redArg(v_a_1296_, v_as_1297_, v_i_boxed_1301_, v_stop_boxed_1302_, v_b_1300_);
lean_dec_ref(v_as_1297_);
return v_res_1303_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___redArg(lean_object* v_as_1304_, size_t v_i_1305_, size_t v_stop_1306_, lean_object* v_b_1307_){
_start:
{
uint8_t v___x_1308_; 
v___x_1308_ = lean_usize_dec_eq(v_i_1305_, v_stop_1306_);
if (v___x_1308_ == 0)
{
size_t v___x_1309_; size_t v___x_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; uint8_t v___x_1313_; 
v___x_1309_ = ((size_t)1ULL);
v___x_1310_ = lean_usize_sub(v_i_1305_, v___x_1309_);
v___x_1311_ = lean_unsigned_to_nat(0u);
v___x_1312_ = lean_array_get_size(v_b_1307_);
v___x_1313_ = lean_nat_dec_lt(v___x_1311_, v___x_1312_);
if (v___x_1313_ == 0)
{
v_i_1305_ = v___x_1310_;
goto _start;
}
else
{
lean_object* v___x_1315_; uint8_t v___x_1316_; 
v___x_1315_ = lean_array_uget_borrowed(v_as_1304_, v___x_1310_);
v___x_1316_ = lean_nat_dec_le(v___x_1312_, v___x_1312_);
if (v___x_1316_ == 0)
{
if (v___x_1313_ == 0)
{
v_i_1305_ = v___x_1310_;
goto _start;
}
else
{
size_t v___x_1318_; size_t v___x_1319_; lean_object* v___x_1320_; 
v___x_1318_ = ((size_t)0ULL);
v___x_1319_ = lean_usize_of_nat(v___x_1312_);
lean_inc_ref(v_b_1307_);
lean_inc(v___x_1315_);
v___x_1320_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___redArg(v___x_1315_, v_b_1307_, v___x_1318_, v___x_1319_, v_b_1307_);
lean_dec_ref(v_b_1307_);
v_i_1305_ = v___x_1310_;
v_b_1307_ = v___x_1320_;
goto _start;
}
}
else
{
size_t v___x_1322_; size_t v___x_1323_; lean_object* v___x_1324_; 
v___x_1322_ = ((size_t)0ULL);
v___x_1323_ = lean_usize_of_nat(v___x_1312_);
lean_inc_ref(v_b_1307_);
lean_inc(v___x_1315_);
v___x_1324_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___redArg(v___x_1315_, v_b_1307_, v___x_1322_, v___x_1323_, v_b_1307_);
lean_dec_ref(v_b_1307_);
v_i_1305_ = v___x_1310_;
v_b_1307_ = v___x_1324_;
goto _start;
}
}
}
else
{
return v_b_1307_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___redArg___boxed(lean_object* v_as_1326_, lean_object* v_i_1327_, lean_object* v_stop_1328_, lean_object* v_b_1329_){
_start:
{
size_t v_i_boxed_1330_; size_t v_stop_boxed_1331_; lean_object* v_res_1332_; 
v_i_boxed_1330_ = lean_unbox_usize(v_i_1327_);
lean_dec(v_i_1327_);
v_stop_boxed_1331_ = lean_unbox_usize(v_stop_1328_);
lean_dec(v_stop_1328_);
v_res_1332_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___redArg(v_as_1326_, v_i_boxed_1330_, v_stop_boxed_1331_, v_b_1329_);
lean_dec_ref(v_as_1326_);
return v_res_1332_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublists_x27_spec__1___redArg(lean_object* v_init_1333_, lean_object* v_l_1334_){
_start:
{
lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; uint8_t v___x_1338_; 
v___x_1335_ = lean_array_mk(v_l_1334_);
v___x_1336_ = lean_array_get_size(v___x_1335_);
v___x_1337_ = lean_unsigned_to_nat(0u);
v___x_1338_ = lean_nat_dec_lt(v___x_1337_, v___x_1336_);
if (v___x_1338_ == 0)
{
lean_dec_ref(v___x_1335_);
return v_init_1333_;
}
else
{
size_t v___x_1339_; size_t v___x_1340_; lean_object* v___x_1341_; 
v___x_1339_ = lean_usize_of_nat(v___x_1336_);
v___x_1340_ = ((size_t)0ULL);
v___x_1341_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___redArg(v___x_1335_, v___x_1339_, v___x_1340_, v_init_1333_);
lean_dec_ref(v___x_1335_);
return v___x_1341_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sublists_x27___redArg(lean_object* v_l_1342_){
_start:
{
lean_object* v___x_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; lean_object* v___x_1347_; 
v___x_1343_ = lean_unsigned_to_nat(1u);
v___x_1344_ = lean_mk_empty_array_with_capacity(v___x_1343_);
lean_dec_ref(v___x_1344_);
v___x_1345_ = ((lean_object*)(lp_batteries_List_initsTR___redArg___closed__0));
v___x_1346_ = lp_batteries_List_foldrTR___at___00List_sublists_x27_spec__1___redArg(v___x_1345_, v_l_1342_);
v___x_1347_ = lean_array_to_list(v___x_1346_);
return v___x_1347_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sublists_x27(lean_object* v_00_u03b1_1348_, lean_object* v_l_1349_){
_start:
{
lean_object* v___x_1350_; 
v___x_1350_ = lp_batteries_List_sublists_x27___redArg(v_l_1349_);
return v___x_1350_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0(lean_object* v_00_u03b1_1351_, lean_object* v_a_1352_, lean_object* v_as_1353_, size_t v_i_1354_, size_t v_stop_1355_, lean_object* v_b_1356_){
_start:
{
lean_object* v___x_1357_; 
v___x_1357_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___redArg(v_a_1352_, v_as_1353_, v_i_1354_, v_stop_1355_, v_b_1356_);
return v___x_1357_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0___boxed(lean_object* v_00_u03b1_1358_, lean_object* v_a_1359_, lean_object* v_as_1360_, lean_object* v_i_1361_, lean_object* v_stop_1362_, lean_object* v_b_1363_){
_start:
{
size_t v_i_boxed_1364_; size_t v_stop_boxed_1365_; lean_object* v_res_1366_; 
v_i_boxed_1364_ = lean_unbox_usize(v_i_1361_);
lean_dec(v_i_1361_);
v_stop_boxed_1365_ = lean_unbox_usize(v_stop_1362_);
lean_dec(v_stop_1362_);
v_res_1366_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublists_x27_spec__0(v_00_u03b1_1358_, v_a_1359_, v_as_1360_, v_i_boxed_1364_, v_stop_boxed_1365_, v_b_1363_);
lean_dec_ref(v_as_1360_);
return v_res_1366_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublists_x27_spec__1(lean_object* v_00_u03b1_1367_, lean_object* v_init_1368_, lean_object* v_l_1369_){
_start:
{
lean_object* v___x_1370_; 
v___x_1370_ = lp_batteries_List_foldrTR___at___00List_sublists_x27_spec__1___redArg(v_init_1368_, v_l_1369_);
return v___x_1370_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1(lean_object* v_00_u03b1_1371_, lean_object* v_as_1372_, size_t v_i_1373_, size_t v_stop_1374_, lean_object* v_b_1375_){
_start:
{
lean_object* v___x_1376_; 
v___x_1376_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___redArg(v_as_1372_, v_i_1373_, v_stop_1374_, v_b_1375_);
return v___x_1376_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1___boxed(lean_object* v_00_u03b1_1377_, lean_object* v_as_1378_, lean_object* v_i_1379_, lean_object* v_stop_1380_, lean_object* v_b_1381_){
_start:
{
size_t v_i_boxed_1382_; size_t v_stop_boxed_1383_; lean_object* v_res_1384_; 
v_i_boxed_1382_ = lean_unbox_usize(v_i_1379_);
lean_dec(v_i_1379_);
v_stop_boxed_1383_ = lean_unbox_usize(v_stop_1380_);
lean_dec(v_stop_1380_);
v_res_1384_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_x27_spec__1_spec__1(v_00_u03b1_1377_, v_as_1378_, v_i_boxed_1382_, v_stop_boxed_1383_, v_b_1381_);
lean_dec_ref(v_as_1378_);
return v_res_1384_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sublists_spec__0___redArg(lean_object* v_a_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_){
_start:
{
if (lean_obj_tag(v_a_1386_) == 0)
{
lean_object* v___x_1388_; 
lean_dec(v_a_1385_);
v___x_1388_ = lean_array_to_list(v_a_1387_);
return v___x_1388_;
}
else
{
lean_object* v_head_1389_; lean_object* v_tail_1390_; lean_object* v___x_1392_; uint8_t v_isShared_1393_; uint8_t v_isSharedCheck_1402_; 
v_head_1389_ = lean_ctor_get(v_a_1386_, 0);
v_tail_1390_ = lean_ctor_get(v_a_1386_, 1);
v_isSharedCheck_1402_ = !lean_is_exclusive(v_a_1386_);
if (v_isSharedCheck_1402_ == 0)
{
v___x_1392_ = v_a_1386_;
v_isShared_1393_ = v_isSharedCheck_1402_;
goto v_resetjp_1391_;
}
else
{
lean_inc(v_tail_1390_);
lean_inc(v_head_1389_);
lean_dec(v_a_1386_);
v___x_1392_ = lean_box(0);
v_isShared_1393_ = v_isSharedCheck_1402_;
goto v_resetjp_1391_;
}
v_resetjp_1391_:
{
lean_object* v___x_1395_; 
lean_inc(v_head_1389_);
lean_inc(v_a_1385_);
if (v_isShared_1393_ == 0)
{
lean_ctor_set(v___x_1392_, 1, v_head_1389_);
lean_ctor_set(v___x_1392_, 0, v_a_1385_);
v___x_1395_ = v___x_1392_;
goto v_reusejp_1394_;
}
else
{
lean_object* v_reuseFailAlloc_1401_; 
v_reuseFailAlloc_1401_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1401_, 0, v_a_1385_);
lean_ctor_set(v_reuseFailAlloc_1401_, 1, v_head_1389_);
v___x_1395_ = v_reuseFailAlloc_1401_;
goto v_reusejp_1394_;
}
v_reusejp_1394_:
{
lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; lean_object* v___x_1399_; 
v___x_1396_ = lean_box(0);
v___x_1397_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1397_, 0, v___x_1395_);
lean_ctor_set(v___x_1397_, 1, v___x_1396_);
v___x_1398_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1398_, 0, v_head_1389_);
lean_ctor_set(v___x_1398_, 1, v___x_1397_);
v___x_1399_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_1387_, v___x_1398_);
v_a_1386_ = v_tail_1390_;
v_a_1387_ = v___x_1399_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___redArg(lean_object* v_as_1403_, size_t v_i_1404_, size_t v_stop_1405_, lean_object* v_b_1406_){
_start:
{
uint8_t v___x_1407_; 
v___x_1407_ = lean_usize_dec_eq(v_i_1404_, v_stop_1405_);
if (v___x_1407_ == 0)
{
size_t v___x_1408_; size_t v___x_1409_; lean_object* v___x_1410_; lean_object* v___x_1411_; lean_object* v___x_1412_; 
v___x_1408_ = ((size_t)1ULL);
v___x_1409_ = lean_usize_sub(v_i_1404_, v___x_1408_);
v___x_1410_ = lean_array_uget_borrowed(v_as_1403_, v___x_1409_);
v___x_1411_ = ((lean_object*)(lp_batteries_List_tailsTR___redArg___closed__0));
lean_inc(v___x_1410_);
v___x_1412_ = lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sublists_spec__0___redArg(v___x_1410_, v_b_1406_, v___x_1411_);
v_i_1404_ = v___x_1409_;
v_b_1406_ = v___x_1412_;
goto _start;
}
else
{
return v_b_1406_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___redArg___boxed(lean_object* v_as_1414_, lean_object* v_i_1415_, lean_object* v_stop_1416_, lean_object* v_b_1417_){
_start:
{
size_t v_i_boxed_1418_; size_t v_stop_boxed_1419_; lean_object* v_res_1420_; 
v_i_boxed_1418_ = lean_unbox_usize(v_i_1415_);
lean_dec(v_i_1415_);
v_stop_boxed_1419_ = lean_unbox_usize(v_stop_1416_);
lean_dec(v_stop_1416_);
v_res_1420_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___redArg(v_as_1414_, v_i_boxed_1418_, v_stop_boxed_1419_, v_b_1417_);
lean_dec_ref(v_as_1414_);
return v_res_1420_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublists_spec__1___redArg(lean_object* v_init_1421_, lean_object* v_l_1422_){
_start:
{
lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; uint8_t v___x_1426_; 
v___x_1423_ = lean_array_mk(v_l_1422_);
v___x_1424_ = lean_array_get_size(v___x_1423_);
v___x_1425_ = lean_unsigned_to_nat(0u);
v___x_1426_ = lean_nat_dec_lt(v___x_1425_, v___x_1424_);
if (v___x_1426_ == 0)
{
lean_dec_ref(v___x_1423_);
return v_init_1421_;
}
else
{
size_t v___x_1427_; size_t v___x_1428_; lean_object* v___x_1429_; 
v___x_1427_ = lean_usize_of_nat(v___x_1424_);
v___x_1428_ = ((size_t)0ULL);
v___x_1429_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___redArg(v___x_1423_, v___x_1427_, v___x_1428_, v_init_1421_);
lean_dec_ref(v___x_1423_);
return v___x_1429_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sublists___redArg(lean_object* v_l_1432_){
_start:
{
lean_object* v___x_1433_; lean_object* v___x_1434_; 
v___x_1433_ = ((lean_object*)(lp_batteries_List_sublists___redArg___closed__0));
v___x_1434_ = lp_batteries_List_foldrTR___at___00List_sublists_spec__1___redArg(v___x_1433_, v_l_1432_);
return v___x_1434_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sublists(lean_object* v_00_u03b1_1435_, lean_object* v_l_1436_){
_start:
{
lean_object* v___x_1437_; 
v___x_1437_ = lp_batteries_List_sublists___redArg(v_l_1436_);
return v___x_1437_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sublists_spec__0(lean_object* v_00_u03b1_1438_, lean_object* v_a_1439_, lean_object* v_a_1440_, lean_object* v_a_1441_){
_start:
{
lean_object* v___x_1442_; 
v___x_1442_ = lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sublists_spec__0___redArg(v_a_1439_, v_a_1440_, v_a_1441_);
return v___x_1442_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublists_spec__1(lean_object* v_00_u03b1_1443_, lean_object* v_init_1444_, lean_object* v_l_1445_){
_start:
{
lean_object* v___x_1446_; 
v___x_1446_ = lp_batteries_List_foldrTR___at___00List_sublists_spec__1___redArg(v_init_1444_, v_l_1445_);
return v___x_1446_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1(lean_object* v_00_u03b1_1447_, lean_object* v_as_1448_, size_t v_i_1449_, size_t v_stop_1450_, lean_object* v_b_1451_){
_start:
{
lean_object* v___x_1452_; 
v___x_1452_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___redArg(v_as_1448_, v_i_1449_, v_stop_1450_, v_b_1451_);
return v___x_1452_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1___boxed(lean_object* v_00_u03b1_1453_, lean_object* v_as_1454_, lean_object* v_i_1455_, lean_object* v_stop_1456_, lean_object* v_b_1457_){
_start:
{
size_t v_i_boxed_1458_; size_t v_stop_boxed_1459_; lean_object* v_res_1460_; 
v_i_boxed_1458_ = lean_unbox_usize(v_i_1455_);
lean_dec(v_i_1455_);
v_stop_boxed_1459_ = lean_unbox_usize(v_stop_1456_);
lean_dec(v_stop_1456_);
v_res_1460_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublists_spec__1_spec__1(v_00_u03b1_1453_, v_as_1454_, v_i_boxed_1458_, v_stop_boxed_1459_, v_b_1457_);
lean_dec_ref(v_as_1454_);
return v_res_1460_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___redArg(lean_object* v_a_1461_, lean_object* v_as_1462_, size_t v_i_1463_, size_t v_stop_1464_, lean_object* v_b_1465_){
_start:
{
uint8_t v___x_1466_; 
v___x_1466_ = lean_usize_dec_eq(v_i_1463_, v_stop_1464_);
if (v___x_1466_ == 0)
{
lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; size_t v___x_1471_; size_t v___x_1472_; 
v___x_1467_ = lean_array_uget_borrowed(v_as_1462_, v_i_1463_);
lean_inc_n(v___x_1467_, 2);
v___x_1468_ = lean_array_push(v_b_1465_, v___x_1467_);
lean_inc(v_a_1461_);
v___x_1469_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1469_, 0, v_a_1461_);
lean_ctor_set(v___x_1469_, 1, v___x_1467_);
v___x_1470_ = lean_array_push(v___x_1468_, v___x_1469_);
v___x_1471_ = ((size_t)1ULL);
v___x_1472_ = lean_usize_add(v_i_1463_, v___x_1471_);
v_i_1463_ = v___x_1472_;
v_b_1465_ = v___x_1470_;
goto _start;
}
else
{
lean_dec(v_a_1461_);
return v_b_1465_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___redArg___boxed(lean_object* v_a_1474_, lean_object* v_as_1475_, lean_object* v_i_1476_, lean_object* v_stop_1477_, lean_object* v_b_1478_){
_start:
{
size_t v_i_boxed_1479_; size_t v_stop_boxed_1480_; lean_object* v_res_1481_; 
v_i_boxed_1479_ = lean_unbox_usize(v_i_1476_);
lean_dec(v_i_1476_);
v_stop_boxed_1480_ = lean_unbox_usize(v_stop_1477_);
lean_dec(v_stop_1477_);
v_res_1481_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___redArg(v_a_1474_, v_as_1475_, v_i_boxed_1479_, v_stop_boxed_1480_, v_b_1478_);
lean_dec_ref(v_as_1475_);
return v_res_1481_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___redArg(lean_object* v_as_1482_, size_t v_i_1483_, size_t v_stop_1484_, lean_object* v_b_1485_){
_start:
{
uint8_t v___x_1486_; 
v___x_1486_ = lean_usize_dec_eq(v_i_1483_, v_stop_1484_);
if (v___x_1486_ == 0)
{
size_t v___x_1487_; size_t v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; uint8_t v___x_1494_; 
v___x_1487_ = ((size_t)1ULL);
v___x_1488_ = lean_usize_sub(v_i_1483_, v___x_1487_);
v___x_1489_ = lean_array_get_size(v_b_1485_);
v___x_1490_ = lean_unsigned_to_nat(2u);
v___x_1491_ = lean_nat_mul(v___x_1489_, v___x_1490_);
v___x_1492_ = lean_mk_empty_array_with_capacity(v___x_1491_);
lean_dec(v___x_1491_);
v___x_1493_ = lean_unsigned_to_nat(0u);
v___x_1494_ = lean_nat_dec_lt(v___x_1493_, v___x_1489_);
if (v___x_1494_ == 0)
{
lean_dec_ref(v_b_1485_);
v_i_1483_ = v___x_1488_;
v_b_1485_ = v___x_1492_;
goto _start;
}
else
{
lean_object* v___x_1496_; uint8_t v___x_1497_; 
v___x_1496_ = lean_array_uget_borrowed(v_as_1482_, v___x_1488_);
v___x_1497_ = lean_nat_dec_le(v___x_1489_, v___x_1489_);
if (v___x_1497_ == 0)
{
if (v___x_1494_ == 0)
{
lean_dec_ref(v_b_1485_);
v_i_1483_ = v___x_1488_;
v_b_1485_ = v___x_1492_;
goto _start;
}
else
{
size_t v___x_1499_; size_t v___x_1500_; lean_object* v___x_1501_; 
v___x_1499_ = ((size_t)0ULL);
v___x_1500_ = lean_usize_of_nat(v___x_1489_);
lean_inc(v___x_1496_);
v___x_1501_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___redArg(v___x_1496_, v_b_1485_, v___x_1499_, v___x_1500_, v___x_1492_);
lean_dec_ref(v_b_1485_);
v_i_1483_ = v___x_1488_;
v_b_1485_ = v___x_1501_;
goto _start;
}
}
else
{
size_t v___x_1503_; size_t v___x_1504_; lean_object* v___x_1505_; 
v___x_1503_ = ((size_t)0ULL);
v___x_1504_ = lean_usize_of_nat(v___x_1489_);
lean_inc(v___x_1496_);
v___x_1505_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___redArg(v___x_1496_, v_b_1485_, v___x_1503_, v___x_1504_, v___x_1492_);
lean_dec_ref(v_b_1485_);
v_i_1483_ = v___x_1488_;
v_b_1485_ = v___x_1505_;
goto _start;
}
}
}
else
{
return v_b_1485_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___redArg___boxed(lean_object* v_as_1507_, lean_object* v_i_1508_, lean_object* v_stop_1509_, lean_object* v_b_1510_){
_start:
{
size_t v_i_boxed_1511_; size_t v_stop_boxed_1512_; lean_object* v_res_1513_; 
v_i_boxed_1511_ = lean_unbox_usize(v_i_1508_);
lean_dec(v_i_1508_);
v_stop_boxed_1512_ = lean_unbox_usize(v_stop_1509_);
lean_dec(v_stop_1509_);
v_res_1513_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___redArg(v_as_1507_, v_i_boxed_1511_, v_stop_boxed_1512_, v_b_1510_);
lean_dec_ref(v_as_1507_);
return v_res_1513_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublistsFast_spec__1___redArg(lean_object* v_init_1514_, lean_object* v_l_1515_){
_start:
{
lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; uint8_t v___x_1519_; 
v___x_1516_ = lean_array_mk(v_l_1515_);
v___x_1517_ = lean_array_get_size(v___x_1516_);
v___x_1518_ = lean_unsigned_to_nat(0u);
v___x_1519_ = lean_nat_dec_lt(v___x_1518_, v___x_1517_);
if (v___x_1519_ == 0)
{
lean_dec_ref(v___x_1516_);
return v_init_1514_;
}
else
{
size_t v___x_1520_; size_t v___x_1521_; lean_object* v___x_1522_; 
v___x_1520_ = lean_usize_of_nat(v___x_1517_);
v___x_1521_ = ((size_t)0ULL);
v___x_1522_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___redArg(v___x_1516_, v___x_1520_, v___x_1521_, v_init_1514_);
lean_dec_ref(v___x_1516_);
return v___x_1522_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sublistsFast___redArg(lean_object* v_l_1523_){
_start:
{
lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; 
v___x_1524_ = lean_unsigned_to_nat(1u);
v___x_1525_ = lean_mk_empty_array_with_capacity(v___x_1524_);
lean_dec_ref(v___x_1525_);
v___x_1526_ = ((lean_object*)(lp_batteries_List_initsTR___redArg___closed__0));
v___x_1527_ = lp_batteries_List_foldrTR___at___00List_sublistsFast_spec__1___redArg(v___x_1526_, v_l_1523_);
v___x_1528_ = lean_array_to_list(v___x_1527_);
return v___x_1528_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sublistsFast(lean_object* v_00_u03b1_1529_, lean_object* v_l_1530_){
_start:
{
lean_object* v___x_1531_; 
v___x_1531_ = lp_batteries_List_sublistsFast___redArg(v_l_1530_);
return v___x_1531_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0(lean_object* v_00_u03b1_1532_, lean_object* v_a_1533_, lean_object* v_as_1534_, size_t v_i_1535_, size_t v_stop_1536_, lean_object* v_b_1537_){
_start:
{
lean_object* v___x_1538_; 
v___x_1538_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___redArg(v_a_1533_, v_as_1534_, v_i_1535_, v_stop_1536_, v_b_1537_);
return v___x_1538_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0___boxed(lean_object* v_00_u03b1_1539_, lean_object* v_a_1540_, lean_object* v_as_1541_, lean_object* v_i_1542_, lean_object* v_stop_1543_, lean_object* v_b_1544_){
_start:
{
size_t v_i_boxed_1545_; size_t v_stop_boxed_1546_; lean_object* v_res_1547_; 
v_i_boxed_1545_ = lean_unbox_usize(v_i_1542_);
lean_dec(v_i_1542_);
v_stop_boxed_1546_ = lean_unbox_usize(v_stop_1543_);
lean_dec(v_stop_1543_);
v_res_1547_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sublistsFast_spec__0(v_00_u03b1_1539_, v_a_1540_, v_as_1541_, v_i_boxed_1545_, v_stop_boxed_1546_, v_b_1544_);
lean_dec_ref(v_as_1541_);
return v_res_1547_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sublistsFast_spec__1(lean_object* v_00_u03b1_1548_, lean_object* v_init_1549_, lean_object* v_l_1550_){
_start:
{
lean_object* v___x_1551_; 
v___x_1551_ = lp_batteries_List_foldrTR___at___00List_sublistsFast_spec__1___redArg(v_init_1549_, v_l_1550_);
return v___x_1551_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1(lean_object* v_00_u03b1_1552_, lean_object* v_as_1553_, size_t v_i_1554_, size_t v_stop_1555_, lean_object* v_b_1556_){
_start:
{
lean_object* v___x_1557_; 
v___x_1557_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___redArg(v_as_1553_, v_i_1554_, v_stop_1555_, v_b_1556_);
return v___x_1557_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1___boxed(lean_object* v_00_u03b1_1558_, lean_object* v_as_1559_, lean_object* v_i_1560_, lean_object* v_stop_1561_, lean_object* v_b_1562_){
_start:
{
size_t v_i_boxed_1563_; size_t v_stop_boxed_1564_; lean_object* v_res_1565_; 
v_i_boxed_1563_ = lean_unbox_usize(v_i_1560_);
lean_dec(v_i_1560_);
v_stop_boxed_1564_ = lean_unbox_usize(v_stop_1561_);
lean_dec(v_stop_1561_);
v_res_1565_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sublistsFast_spec__1_spec__1(v_00_u03b1_1558_, v_as_1559_, v_i_boxed_1563_, v_stop_boxed_1564_, v_b_1562_);
lean_dec_ref(v_as_1559_);
return v_res_1565_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_all_u2082___redArg(lean_object* v_r_1566_, lean_object* v_x_1567_, lean_object* v_x_1568_){
_start:
{
if (lean_obj_tag(v_x_1567_) == 0)
{
lean_dec_ref(v_r_1566_);
if (lean_obj_tag(v_x_1568_) == 0)
{
uint8_t v___x_1569_; 
v___x_1569_ = 1;
return v___x_1569_;
}
else
{
uint8_t v___x_1570_; 
lean_dec(v_x_1568_);
v___x_1570_ = 0;
return v___x_1570_;
}
}
else
{
if (lean_obj_tag(v_x_1568_) == 1)
{
lean_object* v_head_1571_; lean_object* v_tail_1572_; lean_object* v_head_1573_; lean_object* v_tail_1574_; lean_object* v___x_1575_; uint8_t v___x_1576_; 
v_head_1571_ = lean_ctor_get(v_x_1567_, 0);
lean_inc(v_head_1571_);
v_tail_1572_ = lean_ctor_get(v_x_1567_, 1);
lean_inc(v_tail_1572_);
lean_dec_ref_known(v_x_1567_, 2);
v_head_1573_ = lean_ctor_get(v_x_1568_, 0);
lean_inc(v_head_1573_);
v_tail_1574_ = lean_ctor_get(v_x_1568_, 1);
lean_inc(v_tail_1574_);
lean_dec_ref_known(v_x_1568_, 2);
lean_inc_ref(v_r_1566_);
v___x_1575_ = lean_apply_2(v_r_1566_, v_head_1571_, v_head_1573_);
v___x_1576_ = lean_unbox(v___x_1575_);
if (v___x_1576_ == 0)
{
uint8_t v___x_1577_; 
lean_dec(v_tail_1574_);
lean_dec(v_tail_1572_);
lean_dec_ref(v_r_1566_);
v___x_1577_ = lean_unbox(v___x_1575_);
return v___x_1577_;
}
else
{
v_x_1567_ = v_tail_1572_;
v_x_1568_ = v_tail_1574_;
goto _start;
}
}
else
{
uint8_t v___x_1579_; 
lean_dec_ref_known(v_x_1567_, 2);
lean_dec(v_x_1568_);
lean_dec_ref(v_r_1566_);
v___x_1579_ = 0;
return v___x_1579_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_all_u2082___redArg___boxed(lean_object* v_r_1580_, lean_object* v_x_1581_, lean_object* v_x_1582_){
_start:
{
uint8_t v_res_1583_; lean_object* v_r_1584_; 
v_res_1583_ = lp_batteries_List_all_u2082___redArg(v_r_1580_, v_x_1581_, v_x_1582_);
v_r_1584_ = lean_box(v_res_1583_);
return v_r_1584_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_all_u2082(lean_object* v_00_u03b1_1585_, lean_object* v_00_u03b2_1586_, lean_object* v_r_1587_, lean_object* v_x_1588_, lean_object* v_x_1589_){
_start:
{
uint8_t v___x_1590_; 
v___x_1590_ = lp_batteries_List_all_u2082___redArg(v_r_1587_, v_x_1588_, v_x_1589_);
return v___x_1590_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_all_u2082___boxed(lean_object* v_00_u03b1_1591_, lean_object* v_00_u03b2_1592_, lean_object* v_r_1593_, lean_object* v_x_1594_, lean_object* v_x_1595_){
_start:
{
uint8_t v_res_1596_; lean_object* v_r_1597_; 
v_res_1596_ = lp_batteries_List_all_u2082(v_00_u03b1_1591_, v_00_u03b2_1592_, v_r_1593_, v_x_1594_, v_x_1595_);
v_r_1597_ = lean_box(v_res_1596_);
return v_r_1597_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_all_u2082_match__1_splitter___redArg(lean_object* v_x_1598_, lean_object* v_x_1599_, lean_object* v_h__1_1600_, lean_object* v_h__2_1601_, lean_object* v_h__3_1602_){
_start:
{
if (lean_obj_tag(v_x_1598_) == 0)
{
lean_dec(v_h__2_1601_);
if (lean_obj_tag(v_x_1599_) == 0)
{
lean_object* v___x_1603_; lean_object* v___x_1604_; 
lean_dec(v_h__3_1602_);
v___x_1603_ = lean_box(0);
v___x_1604_ = lean_apply_1(v_h__1_1600_, v___x_1603_);
return v___x_1604_;
}
else
{
lean_object* v___x_1605_; 
lean_dec(v_h__1_1600_);
v___x_1605_ = lean_apply_4(v_h__3_1602_, v_x_1598_, v_x_1599_, lean_box(0), lean_box(0));
return v___x_1605_;
}
}
else
{
lean_dec(v_h__1_1600_);
if (lean_obj_tag(v_x_1599_) == 1)
{
lean_object* v_head_1606_; lean_object* v_tail_1607_; lean_object* v_head_1608_; lean_object* v_tail_1609_; lean_object* v___x_1610_; 
lean_dec(v_h__3_1602_);
v_head_1606_ = lean_ctor_get(v_x_1598_, 0);
lean_inc(v_head_1606_);
v_tail_1607_ = lean_ctor_get(v_x_1598_, 1);
lean_inc(v_tail_1607_);
lean_dec_ref_known(v_x_1598_, 2);
v_head_1608_ = lean_ctor_get(v_x_1599_, 0);
lean_inc(v_head_1608_);
v_tail_1609_ = lean_ctor_get(v_x_1599_, 1);
lean_inc(v_tail_1609_);
lean_dec_ref_known(v_x_1599_, 2);
v___x_1610_ = lean_apply_4(v_h__2_1601_, v_head_1606_, v_tail_1607_, v_head_1608_, v_tail_1609_);
return v___x_1610_;
}
else
{
lean_object* v___x_1611_; 
lean_dec(v_h__2_1601_);
v___x_1611_ = lean_apply_4(v_h__3_1602_, v_x_1598_, v_x_1599_, lean_box(0), lean_box(0));
return v___x_1611_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_all_u2082_match__1_splitter(lean_object* v_00_u03b1_1612_, lean_object* v_00_u03b2_1613_, lean_object* v_motive_1614_, lean_object* v_x_1615_, lean_object* v_x_1616_, lean_object* v_h__1_1617_, lean_object* v_h__2_1618_, lean_object* v_h__3_1619_){
_start:
{
if (lean_obj_tag(v_x_1615_) == 0)
{
lean_dec(v_h__2_1618_);
if (lean_obj_tag(v_x_1616_) == 0)
{
lean_object* v___x_1620_; lean_object* v___x_1621_; 
lean_dec(v_h__3_1619_);
v___x_1620_ = lean_box(0);
v___x_1621_ = lean_apply_1(v_h__1_1617_, v___x_1620_);
return v___x_1621_;
}
else
{
lean_object* v___x_1622_; 
lean_dec(v_h__1_1617_);
v___x_1622_ = lean_apply_4(v_h__3_1619_, v_x_1615_, v_x_1616_, lean_box(0), lean_box(0));
return v___x_1622_;
}
}
else
{
lean_dec(v_h__1_1617_);
if (lean_obj_tag(v_x_1616_) == 1)
{
lean_object* v_head_1623_; lean_object* v_tail_1624_; lean_object* v_head_1625_; lean_object* v_tail_1626_; lean_object* v___x_1627_; 
lean_dec(v_h__3_1619_);
v_head_1623_ = lean_ctor_get(v_x_1615_, 0);
lean_inc(v_head_1623_);
v_tail_1624_ = lean_ctor_get(v_x_1615_, 1);
lean_inc(v_tail_1624_);
lean_dec_ref_known(v_x_1615_, 2);
v_head_1625_ = lean_ctor_get(v_x_1616_, 0);
lean_inc(v_head_1625_);
v_tail_1626_ = lean_ctor_get(v_x_1616_, 1);
lean_inc(v_tail_1626_);
lean_dec_ref_known(v_x_1616_, 2);
v___x_1627_ = lean_apply_4(v_h__2_1618_, v_head_1623_, v_tail_1624_, v_head_1625_, v_tail_1626_);
return v___x_1627_;
}
else
{
lean_object* v___x_1628_; 
lean_dec(v_h__2_1618_);
v___x_1628_ = lean_apply_4(v_h__3_1619_, v_x_1615_, v_x_1616_, lean_box(0), lean_box(0));
return v___x_1628_;
}
}
}
}
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableForall_u2082___redArg___lam__0(lean_object* v_inst_1629_, lean_object* v_x1_1630_, lean_object* v_x2_1631_){
_start:
{
lean_object* v___x_1632_; uint8_t v___x_1633_; 
v___x_1632_ = lean_apply_2(v_inst_1629_, v_x1_1630_, v_x2_1631_);
v___x_1633_ = lean_unbox(v___x_1632_);
return v___x_1633_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableForall_u2082___redArg___lam__0___boxed(lean_object* v_inst_1634_, lean_object* v_x1_1635_, lean_object* v_x2_1636_){
_start:
{
uint8_t v_res_1637_; lean_object* v_r_1638_; 
v_res_1637_ = lp_batteries_List_instDecidableForall_u2082___redArg___lam__0(v_inst_1634_, v_x1_1635_, v_x2_1636_);
v_r_1638_ = lean_box(v_res_1637_);
return v_r_1638_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableForall_u2082___redArg(lean_object* v_inst_1639_, lean_object* v_l_u2081_1640_, lean_object* v_l_u2082_1641_){
_start:
{
lean_object* v___f_1642_; uint8_t v___x_1643_; 
v___f_1642_ = lean_alloc_closure((void*)(lp_batteries_List_instDecidableForall_u2082___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1642_, 0, v_inst_1639_);
v___x_1643_ = lp_batteries_List_all_u2082___redArg(v___f_1642_, v_l_u2081_1640_, v_l_u2082_1641_);
return v___x_1643_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableForall_u2082___redArg___boxed(lean_object* v_inst_1644_, lean_object* v_l_u2081_1645_, lean_object* v_l_u2082_1646_){
_start:
{
uint8_t v_res_1647_; lean_object* v_r_1648_; 
v_res_1647_ = lp_batteries_List_instDecidableForall_u2082___redArg(v_inst_1644_, v_l_u2081_1645_, v_l_u2082_1646_);
v_r_1648_ = lean_box(v_res_1647_);
return v_r_1648_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableForall_u2082(lean_object* v_00_u03b1_1649_, lean_object* v_00_u03b2_1650_, lean_object* v_R_1651_, lean_object* v_inst_1652_, lean_object* v_l_u2081_1653_, lean_object* v_l_u2082_1654_){
_start:
{
uint8_t v___x_1655_; 
v___x_1655_ = lp_batteries_List_instDecidableForall_u2082___redArg(v_inst_1652_, v_l_u2081_1653_, v_l_u2082_1654_);
return v___x_1655_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableForall_u2082___boxed(lean_object* v_00_u03b1_1656_, lean_object* v_00_u03b2_1657_, lean_object* v_R_1658_, lean_object* v_inst_1659_, lean_object* v_l_u2081_1660_, lean_object* v_l_u2082_1661_){
_start:
{
uint8_t v_res_1662_; lean_object* v_r_1663_; 
v_res_1662_ = lp_batteries_List_instDecidableForall_u2082(v_00_u03b1_1656_, v_00_u03b2_1657_, v_R_1658_, v_inst_1659_, v_l_u2081_1660_, v_l_u2082_1661_);
v_r_1663_ = lean_box(v_res_1662_);
return v_r_1663_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_transpose_pop___redArg(lean_object* v_old_1664_, lean_object* v_a_1665_){
_start:
{
if (lean_obj_tag(v_a_1665_) == 0)
{
lean_object* v___x_1666_; 
v___x_1666_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1666_, 0, v_old_1664_);
lean_ctor_set(v___x_1666_, 1, v_a_1665_);
return v___x_1666_;
}
else
{
lean_object* v_head_1667_; lean_object* v_tail_1668_; lean_object* v___x_1670_; uint8_t v_isShared_1671_; uint8_t v_isSharedCheck_1676_; 
v_head_1667_ = lean_ctor_get(v_a_1665_, 0);
v_tail_1668_ = lean_ctor_get(v_a_1665_, 1);
v_isSharedCheck_1676_ = !lean_is_exclusive(v_a_1665_);
if (v_isSharedCheck_1676_ == 0)
{
v___x_1670_ = v_a_1665_;
v_isShared_1671_ = v_isSharedCheck_1676_;
goto v_resetjp_1669_;
}
else
{
lean_inc(v_tail_1668_);
lean_inc(v_head_1667_);
lean_dec(v_a_1665_);
v___x_1670_ = lean_box(0);
v_isShared_1671_ = v_isSharedCheck_1676_;
goto v_resetjp_1669_;
}
v_resetjp_1669_:
{
lean_object* v___x_1673_; 
if (v_isShared_1671_ == 0)
{
lean_ctor_set(v___x_1670_, 1, v_old_1664_);
v___x_1673_ = v___x_1670_;
goto v_reusejp_1672_;
}
else
{
lean_object* v_reuseFailAlloc_1675_; 
v_reuseFailAlloc_1675_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1675_, 0, v_head_1667_);
lean_ctor_set(v_reuseFailAlloc_1675_, 1, v_old_1664_);
v___x_1673_ = v_reuseFailAlloc_1675_;
goto v_reusejp_1672_;
}
v_reusejp_1672_:
{
lean_object* v___x_1674_; 
v___x_1674_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1674_, 0, v___x_1673_);
lean_ctor_set(v___x_1674_, 1, v_tail_1668_);
return v___x_1674_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_transpose_pop(lean_object* v_00_u03b1_1677_, lean_object* v_old_1678_, lean_object* v_a_1679_){
_start:
{
lean_object* v___x_1680_; 
v___x_1680_ = lp_batteries_List_transpose_pop___redArg(v_old_1678_, v_a_1679_);
return v___x_1680_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___redArg(size_t v_sz_1681_, size_t v_i_1682_, lean_object* v_bs_1683_, lean_object* v___y_1684_){
_start:
{
uint8_t v___x_1685_; 
v___x_1685_ = lean_usize_dec_lt(v_i_1682_, v_sz_1681_);
if (v___x_1685_ == 0)
{
lean_object* v___x_1686_; 
v___x_1686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1686_, 0, v_bs_1683_);
lean_ctor_set(v___x_1686_, 1, v___y_1684_);
return v___x_1686_;
}
else
{
lean_object* v_v_1687_; lean_object* v___x_1688_; lean_object* v_fst_1689_; lean_object* v_snd_1690_; lean_object* v___x_1691_; lean_object* v_bs_x27_1692_; size_t v___x_1693_; size_t v___x_1694_; lean_object* v___x_1695_; 
v_v_1687_ = lean_array_uget_borrowed(v_bs_1683_, v_i_1682_);
lean_inc(v_v_1687_);
v___x_1688_ = lp_batteries_List_transpose_pop___redArg(v_v_1687_, v___y_1684_);
v_fst_1689_ = lean_ctor_get(v___x_1688_, 0);
lean_inc(v_fst_1689_);
v_snd_1690_ = lean_ctor_get(v___x_1688_, 1);
lean_inc(v_snd_1690_);
lean_dec_ref(v___x_1688_);
v___x_1691_ = lean_unsigned_to_nat(0u);
v_bs_x27_1692_ = lean_array_uset(v_bs_1683_, v_i_1682_, v___x_1691_);
v___x_1693_ = ((size_t)1ULL);
v___x_1694_ = lean_usize_add(v_i_1682_, v___x_1693_);
v___x_1695_ = lean_array_uset(v_bs_x27_1692_, v_i_1682_, v_fst_1689_);
v_i_1682_ = v___x_1694_;
v_bs_1683_ = v___x_1695_;
v___y_1684_ = v_snd_1690_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___redArg___boxed(lean_object* v_sz_1697_, lean_object* v_i_1698_, lean_object* v_bs_1699_, lean_object* v___y_1700_){
_start:
{
size_t v_sz_boxed_1701_; size_t v_i_boxed_1702_; lean_object* v_res_1703_; 
v_sz_boxed_1701_ = lean_unbox_usize(v_sz_1697_);
lean_dec(v_sz_1697_);
v_i_boxed_1702_ = lean_unbox_usize(v_i_1698_);
lean_dec(v_i_1698_);
v_res_1703_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___redArg(v_sz_boxed_1701_, v_i_boxed_1702_, v_bs_1699_, v___y_1700_);
return v_res_1703_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_transpose_go_spec__1___redArg(lean_object* v_x_1704_, lean_object* v_x_1705_){
_start:
{
if (lean_obj_tag(v_x_1705_) == 0)
{
return v_x_1704_;
}
else
{
lean_object* v_head_1706_; lean_object* v_tail_1707_; lean_object* v___x_1709_; uint8_t v_isShared_1710_; uint8_t v_isSharedCheck_1717_; 
v_head_1706_ = lean_ctor_get(v_x_1705_, 0);
v_tail_1707_ = lean_ctor_get(v_x_1705_, 1);
v_isSharedCheck_1717_ = !lean_is_exclusive(v_x_1705_);
if (v_isSharedCheck_1717_ == 0)
{
v___x_1709_ = v_x_1705_;
v_isShared_1710_ = v_isSharedCheck_1717_;
goto v_resetjp_1708_;
}
else
{
lean_inc(v_tail_1707_);
lean_inc(v_head_1706_);
lean_dec(v_x_1705_);
v___x_1709_ = lean_box(0);
v_isShared_1710_ = v_isSharedCheck_1717_;
goto v_resetjp_1708_;
}
v_resetjp_1708_:
{
lean_object* v___x_1711_; lean_object* v___x_1713_; 
v___x_1711_ = lean_box(0);
if (v_isShared_1710_ == 0)
{
lean_ctor_set(v___x_1709_, 1, v___x_1711_);
v___x_1713_ = v___x_1709_;
goto v_reusejp_1712_;
}
else
{
lean_object* v_reuseFailAlloc_1716_; 
v_reuseFailAlloc_1716_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1716_, 0, v_head_1706_);
lean_ctor_set(v_reuseFailAlloc_1716_, 1, v___x_1711_);
v___x_1713_ = v_reuseFailAlloc_1716_;
goto v_reusejp_1712_;
}
v_reusejp_1712_:
{
lean_object* v___x_1714_; 
v___x_1714_ = lean_array_push(v_x_1704_, v___x_1713_);
v_x_1704_ = v___x_1714_;
v_x_1705_ = v_tail_1707_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_transpose_go___redArg(lean_object* v_l_1718_, lean_object* v_acc_1719_){
_start:
{
size_t v_sz_1720_; size_t v___x_1721_; lean_object* v___x_1722_; lean_object* v_fst_1723_; lean_object* v_snd_1724_; lean_object* v___x_1725_; 
v_sz_1720_ = lean_array_size(v_acc_1719_);
v___x_1721_ = ((size_t)0ULL);
v___x_1722_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___redArg(v_sz_1720_, v___x_1721_, v_acc_1719_, v_l_1718_);
v_fst_1723_ = lean_ctor_get(v___x_1722_, 0);
lean_inc(v_fst_1723_);
v_snd_1724_ = lean_ctor_get(v___x_1722_, 1);
lean_inc(v_snd_1724_);
lean_dec_ref(v___x_1722_);
v___x_1725_ = lp_batteries_List_foldl___at___00List_transpose_go_spec__1___redArg(v_fst_1723_, v_snd_1724_);
return v___x_1725_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_transpose_go(lean_object* v_00_u03b1_1726_, lean_object* v_l_1727_, lean_object* v_acc_1728_){
_start:
{
lean_object* v___x_1729_; 
v___x_1729_ = lp_batteries_List_transpose_go___redArg(v_l_1727_, v_acc_1728_);
return v___x_1729_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0(lean_object* v_00_u03b1_1730_, size_t v_sz_1731_, size_t v_i_1732_, lean_object* v_bs_1733_, lean_object* v___y_1734_){
_start:
{
lean_object* v___x_1735_; 
v___x_1735_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___redArg(v_sz_1731_, v_i_1732_, v_bs_1733_, v___y_1734_);
return v___x_1735_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0___boxed(lean_object* v_00_u03b1_1736_, lean_object* v_sz_1737_, lean_object* v_i_1738_, lean_object* v_bs_1739_, lean_object* v___y_1740_){
_start:
{
size_t v_sz_boxed_1741_; size_t v_i_boxed_1742_; lean_object* v_res_1743_; 
v_sz_boxed_1741_ = lean_unbox_usize(v_sz_1737_);
lean_dec(v_sz_1737_);
v_i_boxed_1742_ = lean_unbox_usize(v_i_1738_);
lean_dec(v_i_1738_);
v_res_1743_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00List_transpose_go_spec__0(v_00_u03b1_1736_, v_sz_boxed_1741_, v_i_boxed_1742_, v_bs_1739_, v___y_1740_);
return v_res_1743_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_transpose_go_spec__1(lean_object* v_00_u03b1_1744_, lean_object* v_x_1745_, lean_object* v_x_1746_){
_start:
{
lean_object* v___x_1747_; 
v___x_1747_ = lp_batteries_List_foldl___at___00List_transpose_go_spec__1___redArg(v_x_1745_, v_x_1746_);
return v___x_1747_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___redArg(lean_object* v_as_1748_, size_t v_i_1749_, size_t v_stop_1750_, lean_object* v_b_1751_){
_start:
{
uint8_t v___x_1752_; 
v___x_1752_ = lean_usize_dec_eq(v_i_1749_, v_stop_1750_);
if (v___x_1752_ == 0)
{
size_t v___x_1753_; size_t v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; 
v___x_1753_ = ((size_t)1ULL);
v___x_1754_ = lean_usize_sub(v_i_1749_, v___x_1753_);
v___x_1755_ = lean_array_uget_borrowed(v_as_1748_, v___x_1754_);
lean_inc(v___x_1755_);
v___x_1756_ = lp_batteries_List_transpose_go___redArg(v___x_1755_, v_b_1751_);
v_i_1749_ = v___x_1754_;
v_b_1751_ = v___x_1756_;
goto _start;
}
else
{
return v_b_1751_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___redArg___boxed(lean_object* v_as_1758_, lean_object* v_i_1759_, lean_object* v_stop_1760_, lean_object* v_b_1761_){
_start:
{
size_t v_i_boxed_1762_; size_t v_stop_boxed_1763_; lean_object* v_res_1764_; 
v_i_boxed_1762_ = lean_unbox_usize(v_i_1759_);
lean_dec(v_i_1759_);
v_stop_boxed_1763_ = lean_unbox_usize(v_stop_1760_);
lean_dec(v_stop_1760_);
v_res_1764_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___redArg(v_as_1758_, v_i_boxed_1762_, v_stop_boxed_1763_, v_b_1761_);
lean_dec_ref(v_as_1758_);
return v_res_1764_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_transpose_spec__0___redArg(lean_object* v_init_1765_, lean_object* v_l_1766_){
_start:
{
lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; uint8_t v___x_1770_; 
v___x_1767_ = lean_array_mk(v_l_1766_);
v___x_1768_ = lean_array_get_size(v___x_1767_);
v___x_1769_ = lean_unsigned_to_nat(0u);
v___x_1770_ = lean_nat_dec_lt(v___x_1769_, v___x_1768_);
if (v___x_1770_ == 0)
{
lean_dec_ref(v___x_1767_);
return v_init_1765_;
}
else
{
size_t v___x_1771_; size_t v___x_1772_; lean_object* v___x_1773_; 
v___x_1771_ = lean_usize_of_nat(v___x_1768_);
v___x_1772_ = ((size_t)0ULL);
v___x_1773_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___redArg(v___x_1767_, v___x_1771_, v___x_1772_, v_init_1765_);
lean_dec_ref(v___x_1767_);
return v___x_1773_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_transpose___redArg(lean_object* v_l_1774_){
_start:
{
lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; 
v___x_1775_ = ((lean_object*)(lp_batteries_List_tailsTR___redArg___closed__0));
v___x_1776_ = lp_batteries_List_foldrTR___at___00List_transpose_spec__0___redArg(v___x_1775_, v_l_1774_);
v___x_1777_ = lean_array_to_list(v___x_1776_);
return v___x_1777_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_transpose(lean_object* v_00_u03b1_1778_, lean_object* v_l_1779_){
_start:
{
lean_object* v___x_1780_; 
v___x_1780_ = lp_batteries_List_transpose___redArg(v_l_1779_);
return v___x_1780_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_transpose_spec__0(lean_object* v_00_u03b1_1781_, lean_object* v_init_1782_, lean_object* v_l_1783_){
_start:
{
lean_object* v___x_1784_; 
v___x_1784_ = lp_batteries_List_foldrTR___at___00List_transpose_spec__0___redArg(v_init_1782_, v_l_1783_);
return v___x_1784_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0(lean_object* v_00_u03b1_1785_, lean_object* v_as_1786_, size_t v_i_1787_, size_t v_stop_1788_, lean_object* v_b_1789_){
_start:
{
lean_object* v___x_1790_; 
v___x_1790_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___redArg(v_as_1786_, v_i_1787_, v_stop_1788_, v_b_1789_);
return v___x_1790_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0___boxed(lean_object* v_00_u03b1_1791_, lean_object* v_as_1792_, lean_object* v_i_1793_, lean_object* v_stop_1794_, lean_object* v_b_1795_){
_start:
{
size_t v_i_boxed_1796_; size_t v_stop_boxed_1797_; lean_object* v_res_1798_; 
v_i_boxed_1796_ = lean_unbox_usize(v_i_1793_);
lean_dec(v_i_1793_);
v_stop_boxed_1797_ = lean_unbox_usize(v_stop_1794_);
lean_dec(v_stop_1794_);
v_res_1798_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_transpose_spec__0_spec__0(v_00_u03b1_1791_, v_as_1792_, v_i_boxed_1796_, v_stop_boxed_1797_, v_b_1795_);
lean_dec_ref(v_as_1792_);
return v_res_1798_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_sections_spec__0___redArg(lean_object* v_s_1799_, lean_object* v_a_1800_, lean_object* v_a_1801_){
_start:
{
if (lean_obj_tag(v_a_1800_) == 0)
{
lean_object* v___x_1802_; 
lean_dec(v_s_1799_);
v___x_1802_ = l_List_reverse___redArg(v_a_1801_);
return v___x_1802_;
}
else
{
lean_object* v_head_1803_; lean_object* v_tail_1804_; lean_object* v___x_1806_; uint8_t v_isShared_1807_; uint8_t v_isSharedCheck_1813_; 
v_head_1803_ = lean_ctor_get(v_a_1800_, 0);
v_tail_1804_ = lean_ctor_get(v_a_1800_, 1);
v_isSharedCheck_1813_ = !lean_is_exclusive(v_a_1800_);
if (v_isSharedCheck_1813_ == 0)
{
v___x_1806_ = v_a_1800_;
v_isShared_1807_ = v_isSharedCheck_1813_;
goto v_resetjp_1805_;
}
else
{
lean_inc(v_tail_1804_);
lean_inc(v_head_1803_);
lean_dec(v_a_1800_);
v___x_1806_ = lean_box(0);
v_isShared_1807_ = v_isSharedCheck_1813_;
goto v_resetjp_1805_;
}
v_resetjp_1805_:
{
lean_object* v___x_1809_; 
lean_inc(v_s_1799_);
if (v_isShared_1807_ == 0)
{
lean_ctor_set(v___x_1806_, 1, v_s_1799_);
v___x_1809_ = v___x_1806_;
goto v_reusejp_1808_;
}
else
{
lean_object* v_reuseFailAlloc_1812_; 
v_reuseFailAlloc_1812_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1812_, 0, v_head_1803_);
lean_ctor_set(v_reuseFailAlloc_1812_, 1, v_s_1799_);
v___x_1809_ = v_reuseFailAlloc_1812_;
goto v_reusejp_1808_;
}
v_reusejp_1808_:
{
lean_object* v___x_1810_; 
v___x_1810_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1810_, 0, v___x_1809_);
lean_ctor_set(v___x_1810_, 1, v_a_1801_);
v_a_1800_ = v_tail_1804_;
v_a_1801_ = v___x_1810_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sections_spec__1___redArg(lean_object* v_head_1814_, lean_object* v_a_1815_, lean_object* v_a_1816_){
_start:
{
if (lean_obj_tag(v_a_1815_) == 0)
{
lean_object* v___x_1817_; 
lean_dec(v_head_1814_);
v___x_1817_ = lean_array_to_list(v_a_1816_);
return v___x_1817_;
}
else
{
lean_object* v_head_1818_; lean_object* v_tail_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; 
v_head_1818_ = lean_ctor_get(v_a_1815_, 0);
lean_inc(v_head_1818_);
v_tail_1819_ = lean_ctor_get(v_a_1815_, 1);
lean_inc(v_tail_1819_);
lean_dec_ref_known(v_a_1815_, 2);
v___x_1820_ = lean_box(0);
lean_inc(v_head_1814_);
v___x_1821_ = lp_batteries_List_mapTR_loop___at___00List_sections_spec__0___redArg(v_head_1818_, v_head_1814_, v___x_1820_);
v___x_1822_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_1816_, v___x_1821_);
v_a_1815_ = v_tail_1819_;
v_a_1816_ = v___x_1822_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sections___redArg(lean_object* v_x_1824_){
_start:
{
if (lean_obj_tag(v_x_1824_) == 0)
{
lean_object* v___x_1825_; lean_object* v___x_1826_; 
v___x_1825_ = lean_box(0);
v___x_1826_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1826_, 0, v___x_1825_);
lean_ctor_set(v___x_1826_, 1, v_x_1824_);
return v___x_1826_;
}
else
{
lean_object* v_head_1827_; lean_object* v_tail_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; 
v_head_1827_ = lean_ctor_get(v_x_1824_, 0);
lean_inc(v_head_1827_);
v_tail_1828_ = lean_ctor_get(v_x_1824_, 1);
lean_inc(v_tail_1828_);
lean_dec_ref_known(v_x_1824_, 2);
v___x_1829_ = lp_batteries_List_sections___redArg(v_tail_1828_);
v___x_1830_ = ((lean_object*)(lp_batteries_List_tailsTR___redArg___closed__0));
v___x_1831_ = lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sections_spec__1___redArg(v_head_1827_, v___x_1829_, v___x_1830_);
return v___x_1831_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sections(lean_object* v_00_u03b1_1832_, lean_object* v_x_1833_){
_start:
{
lean_object* v___x_1834_; 
v___x_1834_ = lp_batteries_List_sections___redArg(v_x_1833_);
return v___x_1834_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_sections_spec__0(lean_object* v_00_u03b1_1835_, lean_object* v_s_1836_, lean_object* v_a_1837_, lean_object* v_a_1838_){
_start:
{
lean_object* v___x_1839_; 
v___x_1839_ = lp_batteries_List_mapTR_loop___at___00List_sections_spec__0___redArg(v_s_1836_, v_a_1837_, v_a_1838_);
return v___x_1839_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sections_spec__1(lean_object* v_00_u03b1_1840_, lean_object* v_head_1841_, lean_object* v_a_1842_, lean_object* v_a_1843_){
_start:
{
lean_object* v___x_1844_; 
v___x_1844_ = lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sections_spec__1___redArg(v_head_1841_, v_a_1842_, v_a_1843_);
return v___x_1844_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_sections_match__1_splitter___redArg(lean_object* v_x_1845_, lean_object* v_h__1_1846_, lean_object* v_h__2_1847_){
_start:
{
if (lean_obj_tag(v_x_1845_) == 0)
{
lean_object* v___x_1848_; lean_object* v___x_1849_; 
lean_dec(v_h__2_1847_);
v___x_1848_ = lean_box(0);
v___x_1849_ = lean_apply_1(v_h__1_1846_, v___x_1848_);
return v___x_1849_;
}
else
{
lean_object* v_head_1850_; lean_object* v_tail_1851_; lean_object* v___x_1852_; 
lean_dec(v_h__1_1846_);
v_head_1850_ = lean_ctor_get(v_x_1845_, 0);
lean_inc(v_head_1850_);
v_tail_1851_ = lean_ctor_get(v_x_1845_, 1);
lean_inc(v_tail_1851_);
lean_dec_ref_known(v_x_1845_, 2);
v___x_1852_ = lean_apply_2(v_h__2_1847_, v_head_1850_, v_tail_1851_);
return v___x_1852_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_sections_match__1_splitter(lean_object* v_00_u03b1_1853_, lean_object* v_motive_1854_, lean_object* v_x_1855_, lean_object* v_h__1_1856_, lean_object* v_h__2_1857_){
_start:
{
if (lean_obj_tag(v_x_1855_) == 0)
{
lean_object* v___x_1858_; lean_object* v___x_1859_; 
lean_dec(v_h__2_1857_);
v___x_1858_ = lean_box(0);
v___x_1859_ = lean_apply_1(v_h__1_1856_, v___x_1858_);
return v___x_1859_;
}
else
{
lean_object* v_head_1860_; lean_object* v_tail_1861_; lean_object* v___x_1862_; 
lean_dec(v_h__1_1856_);
v_head_1860_ = lean_ctor_get(v_x_1855_, 0);
lean_inc(v_head_1860_);
v_tail_1861_ = lean_ctor_get(v_x_1855_, 1);
lean_inc(v_tail_1861_);
lean_dec_ref_known(v_x_1855_, 2);
v___x_1862_ = lean_apply_2(v_h__2_1857_, v_head_1860_, v_tail_1861_);
return v___x_1862_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sectionsTR_go_spec__0___redArg(lean_object* v_x2_1863_, lean_object* v_x_1864_, lean_object* v_x_1865_){
_start:
{
if (lean_obj_tag(v_x_1865_) == 0)
{
lean_dec(v_x2_1863_);
return v_x_1864_;
}
else
{
lean_object* v_head_1866_; lean_object* v_tail_1867_; lean_object* v___x_1869_; uint8_t v_isShared_1870_; uint8_t v_isSharedCheck_1876_; 
v_head_1866_ = lean_ctor_get(v_x_1865_, 0);
v_tail_1867_ = lean_ctor_get(v_x_1865_, 1);
v_isSharedCheck_1876_ = !lean_is_exclusive(v_x_1865_);
if (v_isSharedCheck_1876_ == 0)
{
v___x_1869_ = v_x_1865_;
v_isShared_1870_ = v_isSharedCheck_1876_;
goto v_resetjp_1868_;
}
else
{
lean_inc(v_tail_1867_);
lean_inc(v_head_1866_);
lean_dec(v_x_1865_);
v___x_1869_ = lean_box(0);
v_isShared_1870_ = v_isSharedCheck_1876_;
goto v_resetjp_1868_;
}
v_resetjp_1868_:
{
lean_object* v___x_1872_; 
lean_inc(v_x2_1863_);
if (v_isShared_1870_ == 0)
{
lean_ctor_set(v___x_1869_, 1, v_x2_1863_);
v___x_1872_ = v___x_1869_;
goto v_reusejp_1871_;
}
else
{
lean_object* v_reuseFailAlloc_1875_; 
v_reuseFailAlloc_1875_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1875_, 0, v_head_1866_);
lean_ctor_set(v_reuseFailAlloc_1875_, 1, v_x2_1863_);
v___x_1872_ = v_reuseFailAlloc_1875_;
goto v_reusejp_1871_;
}
v_reusejp_1871_:
{
lean_object* v___x_1873_; 
v___x_1873_ = lean_array_push(v_x_1864_, v___x_1872_);
v_x_1864_ = v___x_1873_;
v_x_1865_ = v_tail_1867_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___redArg(lean_object* v_l_1877_, lean_object* v_as_1878_, size_t v_i_1879_, size_t v_stop_1880_, lean_object* v_b_1881_){
_start:
{
uint8_t v___x_1882_; 
v___x_1882_ = lean_usize_dec_eq(v_i_1879_, v_stop_1880_);
if (v___x_1882_ == 0)
{
lean_object* v___x_1883_; lean_object* v___x_1884_; size_t v___x_1885_; size_t v___x_1886_; 
v___x_1883_ = lean_array_uget_borrowed(v_as_1878_, v_i_1879_);
lean_inc(v_l_1877_);
lean_inc(v___x_1883_);
v___x_1884_ = lp_batteries_List_foldl___at___00List_sectionsTR_go_spec__0___redArg(v___x_1883_, v_b_1881_, v_l_1877_);
v___x_1885_ = ((size_t)1ULL);
v___x_1886_ = lean_usize_add(v_i_1879_, v___x_1885_);
v_i_1879_ = v___x_1886_;
v_b_1881_ = v___x_1884_;
goto _start;
}
else
{
lean_dec(v_l_1877_);
return v_b_1881_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___redArg___boxed(lean_object* v_l_1888_, lean_object* v_as_1889_, lean_object* v_i_1890_, lean_object* v_stop_1891_, lean_object* v_b_1892_){
_start:
{
size_t v_i_boxed_1893_; size_t v_stop_boxed_1894_; lean_object* v_res_1895_; 
v_i_boxed_1893_ = lean_unbox_usize(v_i_1890_);
lean_dec(v_i_1890_);
v_stop_boxed_1894_ = lean_unbox_usize(v_stop_1891_);
lean_dec(v_stop_1891_);
v_res_1895_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___redArg(v_l_1888_, v_as_1889_, v_i_boxed_1893_, v_stop_boxed_1894_, v_b_1892_);
lean_dec_ref(v_as_1889_);
return v_res_1895_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR_go___redArg(lean_object* v_l_1896_, lean_object* v_acc_1897_){
_start:
{
lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; uint8_t v___x_1901_; 
v___x_1898_ = lean_unsigned_to_nat(0u);
v___x_1899_ = ((lean_object*)(lp_batteries_List_tailsTR___redArg___closed__0));
v___x_1900_ = lean_array_get_size(v_acc_1897_);
v___x_1901_ = lean_nat_dec_lt(v___x_1898_, v___x_1900_);
if (v___x_1901_ == 0)
{
lean_dec(v_l_1896_);
return v___x_1899_;
}
else
{
uint8_t v___x_1902_; 
v___x_1902_ = lean_nat_dec_le(v___x_1900_, v___x_1900_);
if (v___x_1902_ == 0)
{
if (v___x_1901_ == 0)
{
lean_dec(v_l_1896_);
return v___x_1899_;
}
else
{
size_t v___x_1903_; size_t v___x_1904_; lean_object* v___x_1905_; 
v___x_1903_ = ((size_t)0ULL);
v___x_1904_ = lean_usize_of_nat(v___x_1900_);
v___x_1905_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___redArg(v_l_1896_, v_acc_1897_, v___x_1903_, v___x_1904_, v___x_1899_);
return v___x_1905_;
}
}
else
{
size_t v___x_1906_; size_t v___x_1907_; lean_object* v___x_1908_; 
v___x_1906_ = ((size_t)0ULL);
v___x_1907_ = lean_usize_of_nat(v___x_1900_);
v___x_1908_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___redArg(v_l_1896_, v_acc_1897_, v___x_1906_, v___x_1907_, v___x_1899_);
return v___x_1908_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR_go___redArg___boxed(lean_object* v_l_1909_, lean_object* v_acc_1910_){
_start:
{
lean_object* v_res_1911_; 
v_res_1911_ = lp_batteries_List_sectionsTR_go___redArg(v_l_1909_, v_acc_1910_);
lean_dec_ref(v_acc_1910_);
return v_res_1911_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR_go(lean_object* v_00_u03b1_1912_, lean_object* v_l_1913_, lean_object* v_acc_1914_){
_start:
{
lean_object* v___x_1915_; 
v___x_1915_ = lp_batteries_List_sectionsTR_go___redArg(v_l_1913_, v_acc_1914_);
return v___x_1915_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR_go___boxed(lean_object* v_00_u03b1_1916_, lean_object* v_l_1917_, lean_object* v_acc_1918_){
_start:
{
lean_object* v_res_1919_; 
v_res_1919_ = lp_batteries_List_sectionsTR_go(v_00_u03b1_1916_, v_l_1917_, v_acc_1918_);
lean_dec_ref(v_acc_1918_);
return v_res_1919_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sectionsTR_go_spec__0(lean_object* v_00_u03b1_1920_, lean_object* v_x2_1921_, lean_object* v_x_1922_, lean_object* v_x_1923_){
_start:
{
lean_object* v___x_1924_; 
v___x_1924_ = lp_batteries_List_foldl___at___00List_sectionsTR_go_spec__0___redArg(v_x2_1921_, v_x_1922_, v_x_1923_);
return v___x_1924_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1(lean_object* v_00_u03b1_1925_, lean_object* v_l_1926_, lean_object* v_as_1927_, size_t v_i_1928_, size_t v_stop_1929_, lean_object* v_b_1930_){
_start:
{
lean_object* v___x_1931_; 
v___x_1931_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___redArg(v_l_1926_, v_as_1927_, v_i_1928_, v_stop_1929_, v_b_1930_);
return v___x_1931_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1___boxed(lean_object* v_00_u03b1_1932_, lean_object* v_l_1933_, lean_object* v_as_1934_, lean_object* v_i_1935_, lean_object* v_stop_1936_, lean_object* v_b_1937_){
_start:
{
size_t v_i_boxed_1938_; size_t v_stop_boxed_1939_; lean_object* v_res_1940_; 
v_i_boxed_1938_ = lean_unbox_usize(v_i_1935_);
lean_dec(v_i_1935_);
v_stop_boxed_1939_ = lean_unbox_usize(v_stop_1936_);
lean_dec(v_stop_1936_);
v_res_1940_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00List_sectionsTR_go_spec__1(v_00_u03b1_1932_, v_l_1933_, v_as_1934_, v_i_boxed_1938_, v_stop_boxed_1939_, v_b_1937_);
lean_dec_ref(v_as_1934_);
return v_res_1940_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00List_sectionsTR_spec__0___redArg(lean_object* v_x_1941_){
_start:
{
if (lean_obj_tag(v_x_1941_) == 0)
{
uint8_t v___x_1942_; 
v___x_1942_ = 0;
return v___x_1942_;
}
else
{
lean_object* v_head_1943_; lean_object* v_tail_1944_; uint8_t v___x_1945_; 
v_head_1943_ = lean_ctor_get(v_x_1941_, 0);
v_tail_1944_ = lean_ctor_get(v_x_1941_, 1);
v___x_1945_ = l_List_isEmpty___redArg(v_head_1943_);
if (v___x_1945_ == 0)
{
v_x_1941_ = v_tail_1944_;
goto _start;
}
else
{
return v___x_1945_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00List_sectionsTR_spec__0___redArg___boxed(lean_object* v_x_1947_){
_start:
{
uint8_t v_res_1948_; lean_object* v_r_1949_; 
v_res_1948_ = lp_batteries_List_any___at___00List_sectionsTR_spec__0___redArg(v_x_1947_);
lean_dec(v_x_1947_);
v_r_1949_ = lean_box(v_res_1948_);
return v_r_1949_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___redArg(lean_object* v_as_1950_, size_t v_i_1951_, size_t v_stop_1952_, lean_object* v_b_1953_){
_start:
{
uint8_t v___x_1954_; 
v___x_1954_ = lean_usize_dec_eq(v_i_1951_, v_stop_1952_);
if (v___x_1954_ == 0)
{
size_t v___x_1955_; size_t v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1958_; 
v___x_1955_ = ((size_t)1ULL);
v___x_1956_ = lean_usize_sub(v_i_1951_, v___x_1955_);
v___x_1957_ = lean_array_uget_borrowed(v_as_1950_, v___x_1956_);
lean_inc(v___x_1957_);
v___x_1958_ = lp_batteries_List_sectionsTR_go___redArg(v___x_1957_, v_b_1953_);
lean_dec_ref(v_b_1953_);
v_i_1951_ = v___x_1956_;
v_b_1953_ = v___x_1958_;
goto _start;
}
else
{
return v_b_1953_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___redArg___boxed(lean_object* v_as_1960_, lean_object* v_i_1961_, lean_object* v_stop_1962_, lean_object* v_b_1963_){
_start:
{
size_t v_i_boxed_1964_; size_t v_stop_boxed_1965_; lean_object* v_res_1966_; 
v_i_boxed_1964_ = lean_unbox_usize(v_i_1961_);
lean_dec(v_i_1961_);
v_stop_boxed_1965_ = lean_unbox_usize(v_stop_1962_);
lean_dec(v_stop_1962_);
v_res_1966_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___redArg(v_as_1960_, v_i_boxed_1964_, v_stop_boxed_1965_, v_b_1963_);
lean_dec_ref(v_as_1960_);
return v_res_1966_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sectionsTR_spec__1___redArg(lean_object* v_init_1967_, lean_object* v_l_1968_){
_start:
{
lean_object* v___x_1969_; lean_object* v___x_1970_; lean_object* v___x_1971_; uint8_t v___x_1972_; 
v___x_1969_ = lean_array_mk(v_l_1968_);
v___x_1970_ = lean_array_get_size(v___x_1969_);
v___x_1971_ = lean_unsigned_to_nat(0u);
v___x_1972_ = lean_nat_dec_lt(v___x_1971_, v___x_1970_);
if (v___x_1972_ == 0)
{
lean_dec_ref(v___x_1969_);
return v_init_1967_;
}
else
{
size_t v___x_1973_; size_t v___x_1974_; lean_object* v___x_1975_; 
v___x_1973_ = lean_usize_of_nat(v___x_1970_);
v___x_1974_ = ((size_t)0ULL);
v___x_1975_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___redArg(v___x_1969_, v___x_1973_, v___x_1974_, v_init_1967_);
lean_dec_ref(v___x_1969_);
return v___x_1975_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR___redArg(lean_object* v_L_1976_){
_start:
{
uint8_t v___x_1977_; 
v___x_1977_ = lp_batteries_List_any___at___00List_sectionsTR_spec__0___redArg(v_L_1976_);
if (v___x_1977_ == 0)
{
lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; 
v___x_1978_ = lean_unsigned_to_nat(1u);
v___x_1979_ = lean_mk_empty_array_with_capacity(v___x_1978_);
lean_dec_ref(v___x_1979_);
v___x_1980_ = ((lean_object*)(lp_batteries_List_initsTR___redArg___closed__0));
v___x_1981_ = lp_batteries_List_foldrTR___at___00List_sectionsTR_spec__1___redArg(v___x_1980_, v_L_1976_);
v___x_1982_ = lean_array_to_list(v___x_1981_);
return v___x_1982_;
}
else
{
lean_object* v___x_1983_; 
lean_dec(v_L_1976_);
v___x_1983_ = lean_box(0);
return v___x_1983_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sectionsTR(lean_object* v_00_u03b1_1984_, lean_object* v_L_1985_){
_start:
{
lean_object* v___x_1986_; 
v___x_1986_ = lp_batteries_List_sectionsTR___redArg(v_L_1985_);
return v___x_1986_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_any___at___00List_sectionsTR_spec__0(lean_object* v_00_u03b1_1987_, lean_object* v_x_1988_){
_start:
{
uint8_t v___x_1989_; 
v___x_1989_ = lp_batteries_List_any___at___00List_sectionsTR_spec__0___redArg(v_x_1988_);
return v___x_1989_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_any___at___00List_sectionsTR_spec__0___boxed(lean_object* v_00_u03b1_1990_, lean_object* v_x_1991_){
_start:
{
uint8_t v_res_1992_; lean_object* v_r_1993_; 
v_res_1992_ = lp_batteries_List_any___at___00List_sectionsTR_spec__0(v_00_u03b1_1990_, v_x_1991_);
lean_dec(v_x_1991_);
v_r_1993_ = lean_box(v_res_1992_);
return v_r_1993_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldrTR___at___00List_sectionsTR_spec__1(lean_object* v_00_u03b1_1994_, lean_object* v_init_1995_, lean_object* v_l_1996_){
_start:
{
lean_object* v___x_1997_; 
v___x_1997_ = lp_batteries_List_foldrTR___at___00List_sectionsTR_spec__1___redArg(v_init_1995_, v_l_1996_);
return v___x_1997_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1(lean_object* v_00_u03b1_1998_, lean_object* v_as_1999_, size_t v_i_2000_, size_t v_stop_2001_, lean_object* v_b_2002_){
_start:
{
lean_object* v___x_2003_; 
v___x_2003_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___redArg(v_as_1999_, v_i_2000_, v_stop_2001_, v_b_2002_);
return v___x_2003_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1___boxed(lean_object* v_00_u03b1_2004_, lean_object* v_as_2005_, lean_object* v_i_2006_, lean_object* v_stop_2007_, lean_object* v_b_2008_){
_start:
{
size_t v_i_boxed_2009_; size_t v_stop_boxed_2010_; lean_object* v_res_2011_; 
v_i_boxed_2009_ = lean_unbox_usize(v_i_2006_);
lean_dec(v_i_2006_);
v_stop_boxed_2010_ = lean_unbox_usize(v_stop_2007_);
lean_dec(v_stop_2007_);
v_res_2011_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_sectionsTR_spec__1_spec__1(v_00_u03b1_2004_, v_as_2005_, v_i_boxed_2009_, v_stop_boxed_2010_, v_b_2008_);
lean_dec_ref(v_as_2005_);
return v_res_2011_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_any_match__1_splitter___redArg(lean_object* v_x_2012_, lean_object* v_x_2013_, lean_object* v_h__1_2014_, lean_object* v_h__2_2015_){
_start:
{
if (lean_obj_tag(v_x_2012_) == 0)
{
lean_object* v___x_2016_; 
lean_dec(v_h__2_2015_);
v___x_2016_ = lean_apply_1(v_h__1_2014_, v_x_2013_);
return v___x_2016_;
}
else
{
lean_object* v_head_2017_; lean_object* v_tail_2018_; lean_object* v___x_2019_; 
lean_dec(v_h__1_2014_);
v_head_2017_ = lean_ctor_get(v_x_2012_, 0);
lean_inc(v_head_2017_);
v_tail_2018_ = lean_ctor_get(v_x_2012_, 1);
lean_inc(v_tail_2018_);
lean_dec_ref_known(v_x_2012_, 2);
v___x_2019_ = lean_apply_3(v_h__2_2015_, v_head_2017_, v_tail_2018_, v_x_2013_);
return v___x_2019_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_any_match__1_splitter(lean_object* v_00_u03b1_2020_, lean_object* v_motive_2021_, lean_object* v_x_2022_, lean_object* v_x_2023_, lean_object* v_h__1_2024_, lean_object* v_h__2_2025_){
_start:
{
if (lean_obj_tag(v_x_2022_) == 0)
{
lean_object* v___x_2026_; 
lean_dec(v_h__2_2025_);
v___x_2026_ = lean_apply_1(v_h__1_2024_, v_x_2023_);
return v___x_2026_;
}
else
{
lean_object* v_head_2027_; lean_object* v_tail_2028_; lean_object* v___x_2029_; 
lean_dec(v_h__1_2024_);
v_head_2027_ = lean_ctor_get(v_x_2022_, 0);
lean_inc(v_head_2027_);
v_tail_2028_ = lean_ctor_get(v_x_2022_, 1);
lean_inc(v_tail_2028_);
lean_dec_ref_known(v_x_2022_, 2);
v___x_2029_ = lean_apply_3(v_h__2_2025_, v_head_2027_, v_tail_2028_, v_x_2023_);
return v___x_2029_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_extractP_go___redArg(lean_object* v_p_2030_, lean_object* v_l_2031_, lean_object* v_a_2032_, lean_object* v_a_2033_){
_start:
{
if (lean_obj_tag(v_a_2032_) == 0)
{
lean_object* v___x_2034_; lean_object* v___x_2035_; 
lean_dec_ref(v_a_2033_);
lean_dec_ref(v_p_2030_);
v___x_2034_ = lean_box(0);
v___x_2035_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2035_, 0, v___x_2034_);
lean_ctor_set(v___x_2035_, 1, v_l_2031_);
return v___x_2035_;
}
else
{
lean_object* v_head_2036_; lean_object* v_tail_2037_; lean_object* v___x_2039_; uint8_t v_isShared_2040_; uint8_t v_isSharedCheck_2058_; 
v_head_2036_ = lean_ctor_get(v_a_2032_, 0);
v_tail_2037_ = lean_ctor_get(v_a_2032_, 1);
v_isSharedCheck_2058_ = !lean_is_exclusive(v_a_2032_);
if (v_isSharedCheck_2058_ == 0)
{
v___x_2039_ = v_a_2032_;
v_isShared_2040_ = v_isSharedCheck_2058_;
goto v_resetjp_2038_;
}
else
{
lean_inc(v_tail_2037_);
lean_inc(v_head_2036_);
lean_dec(v_a_2032_);
v___x_2039_ = lean_box(0);
v_isShared_2040_ = v_isSharedCheck_2058_;
goto v_resetjp_2038_;
}
v_resetjp_2038_:
{
lean_object* v___x_2041_; uint8_t v___x_2042_; 
lean_inc_ref(v_p_2030_);
lean_inc(v_head_2036_);
v___x_2041_ = lean_apply_1(v_p_2030_, v_head_2036_);
v___x_2042_ = lean_unbox(v___x_2041_);
if (v___x_2042_ == 0)
{
lean_object* v___x_2043_; 
lean_del_object(v___x_2039_);
v___x_2043_ = lean_array_push(v_a_2033_, v_head_2036_);
v_a_2032_ = v_tail_2037_;
v_a_2033_ = v___x_2043_;
goto _start;
}
else
{
lean_object* v___x_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; uint8_t v___x_2048_; 
lean_dec(v_l_2031_);
lean_dec_ref(v_p_2030_);
v___x_2045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2045_, 0, v_head_2036_);
v___x_2046_ = lean_array_get_size(v_a_2033_);
v___x_2047_ = lean_unsigned_to_nat(0u);
v___x_2048_ = lean_nat_dec_lt(v___x_2047_, v___x_2046_);
if (v___x_2048_ == 0)
{
lean_object* v___x_2050_; 
lean_dec_ref(v_a_2033_);
if (v_isShared_2040_ == 0)
{
lean_ctor_set_tag(v___x_2039_, 0);
lean_ctor_set(v___x_2039_, 0, v___x_2045_);
v___x_2050_ = v___x_2039_;
goto v_reusejp_2049_;
}
else
{
lean_object* v_reuseFailAlloc_2051_; 
v_reuseFailAlloc_2051_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2051_, 0, v___x_2045_);
lean_ctor_set(v_reuseFailAlloc_2051_, 1, v_tail_2037_);
v___x_2050_ = v_reuseFailAlloc_2051_;
goto v_reusejp_2049_;
}
v_reusejp_2049_:
{
return v___x_2050_;
}
}
else
{
size_t v___x_2052_; size_t v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2056_; 
v___x_2052_ = lean_usize_of_nat(v___x_2046_);
v___x_2053_ = ((size_t)0ULL);
v___x_2054_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg(v_a_2033_, v___x_2052_, v___x_2053_, v_tail_2037_);
lean_dec_ref(v_a_2033_);
if (v_isShared_2040_ == 0)
{
lean_ctor_set_tag(v___x_2039_, 0);
lean_ctor_set(v___x_2039_, 1, v___x_2054_);
lean_ctor_set(v___x_2039_, 0, v___x_2045_);
v___x_2056_ = v___x_2039_;
goto v_reusejp_2055_;
}
else
{
lean_object* v_reuseFailAlloc_2057_; 
v_reuseFailAlloc_2057_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2057_, 0, v___x_2045_);
lean_ctor_set(v_reuseFailAlloc_2057_, 1, v___x_2054_);
v___x_2056_ = v_reuseFailAlloc_2057_;
goto v_reusejp_2055_;
}
v_reusejp_2055_:
{
return v___x_2056_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_extractP_go(lean_object* v_00_u03b1_2059_, lean_object* v_p_2060_, lean_object* v_l_2061_, lean_object* v_a_2062_, lean_object* v_a_2063_){
_start:
{
lean_object* v___x_2064_; 
v___x_2064_ = lp_batteries_List_extractP_go___redArg(v_p_2060_, v_l_2061_, v_a_2062_, v_a_2063_);
return v___x_2064_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_extractP___redArg(lean_object* v_p_2065_, lean_object* v_l_2066_){
_start:
{
lean_object* v___x_2067_; lean_object* v___x_2068_; 
v___x_2067_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_l_2066_);
v___x_2068_ = lp_batteries_List_extractP_go___redArg(v_p_2065_, v_l_2066_, v_l_2066_, v___x_2067_);
return v___x_2068_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_extractP(lean_object* v_00_u03b1_2069_, lean_object* v_p_2070_, lean_object* v_l_2071_){
_start:
{
lean_object* v___x_2072_; 
v___x_2072_ = lp_batteries_List_extractP___redArg(v_p_2070_, v_l_2071_);
return v___x_2072_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_revzip___redArg(lean_object* v_l_2073_){
_start:
{
lean_object* v___x_2074_; lean_object* v___x_2075_; 
lean_inc(v_l_2073_);
v___x_2074_ = l_List_reverse___redArg(v_l_2073_);
v___x_2075_ = l_List_zipWith___at___00List_zip_spec__0(lean_box(0), lean_box(0), v_l_2073_, v___x_2074_);
return v___x_2075_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_revzip(lean_object* v_00_u03b1_2076_, lean_object* v_l_2077_){
_start:
{
lean_object* v___x_2078_; 
v___x_2078_ = lp_batteries_List_revzip___redArg(v_l_2077_);
return v___x_2078_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_product_spec__0___redArg(lean_object* v_a_2079_, lean_object* v_a_2080_, lean_object* v_a_2081_){
_start:
{
if (lean_obj_tag(v_a_2080_) == 0)
{
lean_object* v___x_2082_; 
lean_dec(v_a_2079_);
v___x_2082_ = l_List_reverse___redArg(v_a_2081_);
return v___x_2082_;
}
else
{
lean_object* v_head_2083_; lean_object* v_tail_2084_; lean_object* v___x_2086_; uint8_t v_isShared_2087_; uint8_t v_isSharedCheck_2093_; 
v_head_2083_ = lean_ctor_get(v_a_2080_, 0);
v_tail_2084_ = lean_ctor_get(v_a_2080_, 1);
v_isSharedCheck_2093_ = !lean_is_exclusive(v_a_2080_);
if (v_isSharedCheck_2093_ == 0)
{
v___x_2086_ = v_a_2080_;
v_isShared_2087_ = v_isSharedCheck_2093_;
goto v_resetjp_2085_;
}
else
{
lean_inc(v_tail_2084_);
lean_inc(v_head_2083_);
lean_dec(v_a_2080_);
v___x_2086_ = lean_box(0);
v_isShared_2087_ = v_isSharedCheck_2093_;
goto v_resetjp_2085_;
}
v_resetjp_2085_:
{
lean_object* v___x_2088_; lean_object* v___x_2090_; 
lean_inc(v_a_2079_);
v___x_2088_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2088_, 0, v_a_2079_);
lean_ctor_set(v___x_2088_, 1, v_head_2083_);
if (v_isShared_2087_ == 0)
{
lean_ctor_set(v___x_2086_, 1, v_a_2081_);
lean_ctor_set(v___x_2086_, 0, v___x_2088_);
v___x_2090_ = v___x_2086_;
goto v_reusejp_2089_;
}
else
{
lean_object* v_reuseFailAlloc_2092_; 
v_reuseFailAlloc_2092_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2092_, 0, v___x_2088_);
lean_ctor_set(v_reuseFailAlloc_2092_, 1, v_a_2081_);
v___x_2090_ = v_reuseFailAlloc_2092_;
goto v_reusejp_2089_;
}
v_reusejp_2089_:
{
v_a_2080_ = v_tail_2084_;
v_a_2081_ = v___x_2090_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_product_spec__1___redArg(lean_object* v_l_u2082_2094_, lean_object* v_a_2095_, lean_object* v_a_2096_){
_start:
{
if (lean_obj_tag(v_a_2095_) == 0)
{
lean_object* v___x_2097_; 
lean_dec(v_l_u2082_2094_);
v___x_2097_ = lean_array_to_list(v_a_2096_);
return v___x_2097_;
}
else
{
lean_object* v_head_2098_; lean_object* v_tail_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; lean_object* v___x_2102_; 
v_head_2098_ = lean_ctor_get(v_a_2095_, 0);
lean_inc(v_head_2098_);
v_tail_2099_ = lean_ctor_get(v_a_2095_, 1);
lean_inc(v_tail_2099_);
lean_dec_ref_known(v_a_2095_, 2);
v___x_2100_ = lean_box(0);
lean_inc(v_l_u2082_2094_);
v___x_2101_ = lp_batteries_List_mapTR_loop___at___00List_product_spec__0___redArg(v_head_2098_, v_l_u2082_2094_, v___x_2100_);
v___x_2102_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_2096_, v___x_2101_);
v_a_2095_ = v_tail_2099_;
v_a_2096_ = v___x_2102_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_product___redArg(lean_object* v_l_u2081_2106_, lean_object* v_l_u2082_2107_){
_start:
{
lean_object* v___x_2108_; lean_object* v___x_2109_; 
v___x_2108_ = ((lean_object*)(lp_batteries_List_product___redArg___closed__0));
v___x_2109_ = lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_product_spec__1___redArg(v_l_u2082_2107_, v_l_u2081_2106_, v___x_2108_);
return v___x_2109_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_product(lean_object* v_00_u03b1_2110_, lean_object* v_00_u03b2_2111_, lean_object* v_l_u2081_2112_, lean_object* v_l_u2082_2113_){
_start:
{
lean_object* v___x_2114_; 
v___x_2114_ = lp_batteries_List_product___redArg(v_l_u2081_2112_, v_l_u2082_2113_);
return v___x_2114_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_product_spec__0(lean_object* v_00_u03b2_2115_, lean_object* v_00_u03b1_2116_, lean_object* v_a_2117_, lean_object* v_a_2118_, lean_object* v_a_2119_){
_start:
{
lean_object* v___x_2120_; 
v___x_2120_ = lp_batteries_List_mapTR_loop___at___00List_product_spec__0___redArg(v_a_2117_, v_a_2118_, v_a_2119_);
return v___x_2120_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_product_spec__1(lean_object* v_00_u03b1_2121_, lean_object* v_00_u03b2_2122_, lean_object* v_l_u2082_2123_, lean_object* v_a_2124_, lean_object* v_a_2125_){
_start:
{
lean_object* v___x_2126_; 
v___x_2126_ = lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_product_spec__1___redArg(v_l_u2082_2123_, v_a_2124_, v_a_2125_);
return v___x_2126_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_productTR_spec__0___redArg(lean_object* v_a_2127_, lean_object* v_x_2128_, lean_object* v_x_2129_){
_start:
{
if (lean_obj_tag(v_x_2129_) == 0)
{
lean_dec(v_a_2127_);
return v_x_2128_;
}
else
{
lean_object* v_head_2130_; lean_object* v_tail_2131_; lean_object* v___x_2133_; uint8_t v_isShared_2134_; uint8_t v_isSharedCheck_2140_; 
v_head_2130_ = lean_ctor_get(v_x_2129_, 0);
v_tail_2131_ = lean_ctor_get(v_x_2129_, 1);
v_isSharedCheck_2140_ = !lean_is_exclusive(v_x_2129_);
if (v_isSharedCheck_2140_ == 0)
{
v___x_2133_ = v_x_2129_;
v_isShared_2134_ = v_isSharedCheck_2140_;
goto v_resetjp_2132_;
}
else
{
lean_inc(v_tail_2131_);
lean_inc(v_head_2130_);
lean_dec(v_x_2129_);
v___x_2133_ = lean_box(0);
v_isShared_2134_ = v_isSharedCheck_2140_;
goto v_resetjp_2132_;
}
v_resetjp_2132_:
{
lean_object* v___x_2136_; 
lean_inc(v_a_2127_);
if (v_isShared_2134_ == 0)
{
lean_ctor_set_tag(v___x_2133_, 0);
lean_ctor_set(v___x_2133_, 1, v_head_2130_);
lean_ctor_set(v___x_2133_, 0, v_a_2127_);
v___x_2136_ = v___x_2133_;
goto v_reusejp_2135_;
}
else
{
lean_object* v_reuseFailAlloc_2139_; 
v_reuseFailAlloc_2139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2139_, 0, v_a_2127_);
lean_ctor_set(v_reuseFailAlloc_2139_, 1, v_head_2130_);
v___x_2136_ = v_reuseFailAlloc_2139_;
goto v_reusejp_2135_;
}
v_reusejp_2135_:
{
lean_object* v___x_2137_; 
v___x_2137_ = lean_array_push(v_x_2128_, v___x_2136_);
v_x_2128_ = v___x_2137_;
v_x_2129_ = v_tail_2131_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_productTR_spec__1___redArg(lean_object* v_l_u2082_2141_, lean_object* v_x_2142_, lean_object* v_x_2143_){
_start:
{
if (lean_obj_tag(v_x_2143_) == 0)
{
lean_dec(v_l_u2082_2141_);
return v_x_2142_;
}
else
{
lean_object* v_head_2144_; lean_object* v_tail_2145_; lean_object* v___x_2146_; 
v_head_2144_ = lean_ctor_get(v_x_2143_, 0);
lean_inc(v_head_2144_);
v_tail_2145_ = lean_ctor_get(v_x_2143_, 1);
lean_inc(v_tail_2145_);
lean_dec_ref_known(v_x_2143_, 2);
lean_inc(v_l_u2082_2141_);
v___x_2146_ = lp_batteries_List_foldl___at___00List_productTR_spec__0___redArg(v_head_2144_, v_x_2142_, v_l_u2082_2141_);
v_x_2142_ = v___x_2146_;
v_x_2143_ = v_tail_2145_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_productTR___redArg(lean_object* v_l_u2081_2148_, lean_object* v_l_u2082_2149_){
_start:
{
lean_object* v___x_2150_; lean_object* v___x_2151_; lean_object* v___x_2152_; 
v___x_2150_ = ((lean_object*)(lp_batteries_List_product___redArg___closed__0));
v___x_2151_ = lp_batteries_List_foldl___at___00List_productTR_spec__1___redArg(v_l_u2082_2149_, v___x_2150_, v_l_u2081_2148_);
v___x_2152_ = lean_array_to_list(v___x_2151_);
return v___x_2152_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_productTR(lean_object* v_00_u03b1_2153_, lean_object* v_00_u03b2_2154_, lean_object* v_l_u2081_2155_, lean_object* v_l_u2082_2156_){
_start:
{
lean_object* v___x_2157_; 
v___x_2157_ = lp_batteries_List_productTR___redArg(v_l_u2081_2155_, v_l_u2082_2156_);
return v___x_2157_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_productTR_spec__0(lean_object* v_00_u03b1_2158_, lean_object* v_00_u03b2_2159_, lean_object* v_a_2160_, lean_object* v_x_2161_, lean_object* v_x_2162_){
_start:
{
lean_object* v___x_2163_; 
v___x_2163_ = lp_batteries_List_foldl___at___00List_productTR_spec__0___redArg(v_a_2160_, v_x_2161_, v_x_2162_);
return v___x_2163_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_productTR_spec__1(lean_object* v_00_u03b1_2164_, lean_object* v_00_u03b2_2165_, lean_object* v_l_u2082_2166_, lean_object* v_x_2167_, lean_object* v_x_2168_){
_start:
{
lean_object* v___x_2169_; 
v___x_2169_ = lp_batteries_List_foldl___at___00List_productTR_spec__1___redArg(v_l_u2082_2166_, v_x_2167_, v_x_2168_);
return v___x_2169_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_sigma_spec__0___redArg(lean_object* v_a_2170_, lean_object* v_a_2171_, lean_object* v_a_2172_){
_start:
{
if (lean_obj_tag(v_a_2171_) == 0)
{
lean_object* v___x_2173_; 
lean_dec(v_a_2170_);
v___x_2173_ = l_List_reverse___redArg(v_a_2172_);
return v___x_2173_;
}
else
{
lean_object* v_head_2174_; lean_object* v_tail_2175_; lean_object* v___x_2177_; uint8_t v_isShared_2178_; uint8_t v_isSharedCheck_2184_; 
v_head_2174_ = lean_ctor_get(v_a_2171_, 0);
v_tail_2175_ = lean_ctor_get(v_a_2171_, 1);
v_isSharedCheck_2184_ = !lean_is_exclusive(v_a_2171_);
if (v_isSharedCheck_2184_ == 0)
{
v___x_2177_ = v_a_2171_;
v_isShared_2178_ = v_isSharedCheck_2184_;
goto v_resetjp_2176_;
}
else
{
lean_inc(v_tail_2175_);
lean_inc(v_head_2174_);
lean_dec(v_a_2171_);
v___x_2177_ = lean_box(0);
v_isShared_2178_ = v_isSharedCheck_2184_;
goto v_resetjp_2176_;
}
v_resetjp_2176_:
{
lean_object* v___x_2179_; lean_object* v___x_2181_; 
lean_inc(v_a_2170_);
v___x_2179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2179_, 0, v_a_2170_);
lean_ctor_set(v___x_2179_, 1, v_head_2174_);
if (v_isShared_2178_ == 0)
{
lean_ctor_set(v___x_2177_, 1, v_a_2172_);
lean_ctor_set(v___x_2177_, 0, v___x_2179_);
v___x_2181_ = v___x_2177_;
goto v_reusejp_2180_;
}
else
{
lean_object* v_reuseFailAlloc_2183_; 
v_reuseFailAlloc_2183_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2183_, 0, v___x_2179_);
lean_ctor_set(v_reuseFailAlloc_2183_, 1, v_a_2172_);
v___x_2181_ = v_reuseFailAlloc_2183_;
goto v_reusejp_2180_;
}
v_reusejp_2180_:
{
v_a_2171_ = v_tail_2175_;
v_a_2172_ = v___x_2181_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sigma_spec__1___redArg(lean_object* v_l_u2082_2185_, lean_object* v_a_2186_, lean_object* v_a_2187_){
_start:
{
if (lean_obj_tag(v_a_2186_) == 0)
{
lean_object* v___x_2188_; 
lean_dec_ref(v_l_u2082_2185_);
v___x_2188_ = lean_array_to_list(v_a_2187_);
return v___x_2188_;
}
else
{
lean_object* v_head_2189_; lean_object* v_tail_2190_; lean_object* v___x_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; 
v_head_2189_ = lean_ctor_get(v_a_2186_, 0);
lean_inc_n(v_head_2189_, 2);
v_tail_2190_ = lean_ctor_get(v_a_2186_, 1);
lean_inc(v_tail_2190_);
lean_dec_ref_known(v_a_2186_, 2);
lean_inc_ref(v_l_u2082_2185_);
v___x_2191_ = lean_apply_1(v_l_u2082_2185_, v_head_2189_);
v___x_2192_ = lean_box(0);
v___x_2193_ = lp_batteries_List_mapTR_loop___at___00List_sigma_spec__0___redArg(v_head_2189_, v___x_2191_, v___x_2192_);
v___x_2194_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_2187_, v___x_2193_);
v_a_2186_ = v_tail_2190_;
v_a_2187_ = v___x_2194_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sigma___redArg(lean_object* v_l_u2081_2198_, lean_object* v_l_u2082_2199_){
_start:
{
lean_object* v___x_2200_; lean_object* v___x_2201_; 
v___x_2200_ = ((lean_object*)(lp_batteries_List_sigma___redArg___closed__0));
v___x_2201_ = lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sigma_spec__1___redArg(v_l_u2082_2199_, v_l_u2081_2198_, v___x_2200_);
return v___x_2201_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sigma(lean_object* v_00_u03b1_2202_, lean_object* v_00_u03c3_2203_, lean_object* v_l_u2081_2204_, lean_object* v_l_u2082_2205_){
_start:
{
lean_object* v___x_2206_; 
v___x_2206_ = lp_batteries_List_sigma___redArg(v_l_u2081_2204_, v_l_u2082_2205_);
return v___x_2206_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_sigma_spec__0(lean_object* v_00_u03b1_2207_, lean_object* v_00_u03c3_2208_, lean_object* v_a_2209_, lean_object* v_a_2210_, lean_object* v_a_2211_){
_start:
{
lean_object* v___x_2212_; 
v___x_2212_ = lp_batteries_List_mapTR_loop___at___00List_sigma_spec__0___redArg(v_a_2209_, v_a_2210_, v_a_2211_);
return v___x_2212_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sigma_spec__1(lean_object* v_00_u03b1_2213_, lean_object* v_00_u03c3_2214_, lean_object* v_l_u2082_2215_, lean_object* v_a_2216_, lean_object* v_a_2217_){
_start:
{
lean_object* v___x_2218_; 
v___x_2218_ = lp_batteries___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00List_sigma_spec__1___redArg(v_l_u2082_2215_, v_a_2216_, v_a_2217_);
return v___x_2218_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sigmaTR_spec__0___redArg(lean_object* v_a_2219_, lean_object* v_x_2220_, lean_object* v_x_2221_){
_start:
{
if (lean_obj_tag(v_x_2221_) == 0)
{
lean_dec(v_a_2219_);
return v_x_2220_;
}
else
{
lean_object* v_head_2222_; lean_object* v_tail_2223_; lean_object* v___x_2225_; uint8_t v_isShared_2226_; uint8_t v_isSharedCheck_2232_; 
v_head_2222_ = lean_ctor_get(v_x_2221_, 0);
v_tail_2223_ = lean_ctor_get(v_x_2221_, 1);
v_isSharedCheck_2232_ = !lean_is_exclusive(v_x_2221_);
if (v_isSharedCheck_2232_ == 0)
{
v___x_2225_ = v_x_2221_;
v_isShared_2226_ = v_isSharedCheck_2232_;
goto v_resetjp_2224_;
}
else
{
lean_inc(v_tail_2223_);
lean_inc(v_head_2222_);
lean_dec(v_x_2221_);
v___x_2225_ = lean_box(0);
v_isShared_2226_ = v_isSharedCheck_2232_;
goto v_resetjp_2224_;
}
v_resetjp_2224_:
{
lean_object* v___x_2228_; 
lean_inc(v_a_2219_);
if (v_isShared_2226_ == 0)
{
lean_ctor_set_tag(v___x_2225_, 0);
lean_ctor_set(v___x_2225_, 1, v_head_2222_);
lean_ctor_set(v___x_2225_, 0, v_a_2219_);
v___x_2228_ = v___x_2225_;
goto v_reusejp_2227_;
}
else
{
lean_object* v_reuseFailAlloc_2231_; 
v_reuseFailAlloc_2231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2231_, 0, v_a_2219_);
lean_ctor_set(v_reuseFailAlloc_2231_, 1, v_head_2222_);
v___x_2228_ = v_reuseFailAlloc_2231_;
goto v_reusejp_2227_;
}
v_reusejp_2227_:
{
lean_object* v___x_2229_; 
v___x_2229_ = lean_array_push(v_x_2220_, v___x_2228_);
v_x_2220_ = v___x_2229_;
v_x_2221_ = v_tail_2223_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sigmaTR_spec__1___redArg(lean_object* v_l_u2082_2233_, lean_object* v_x_2234_, lean_object* v_x_2235_){
_start:
{
if (lean_obj_tag(v_x_2235_) == 0)
{
lean_dec_ref(v_l_u2082_2233_);
return v_x_2234_;
}
else
{
lean_object* v_head_2236_; lean_object* v_tail_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; 
v_head_2236_ = lean_ctor_get(v_x_2235_, 0);
lean_inc_n(v_head_2236_, 2);
v_tail_2237_ = lean_ctor_get(v_x_2235_, 1);
lean_inc(v_tail_2237_);
lean_dec_ref_known(v_x_2235_, 2);
lean_inc_ref(v_l_u2082_2233_);
v___x_2238_ = lean_apply_1(v_l_u2082_2233_, v_head_2236_);
v___x_2239_ = lp_batteries_List_foldl___at___00List_sigmaTR_spec__0___redArg(v_head_2236_, v_x_2234_, v___x_2238_);
v_x_2234_ = v___x_2239_;
v_x_2235_ = v_tail_2237_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sigmaTR___redArg(lean_object* v_l_u2081_2241_, lean_object* v_l_u2082_2242_){
_start:
{
lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; 
v___x_2243_ = ((lean_object*)(lp_batteries_List_sigma___redArg___closed__0));
v___x_2244_ = lp_batteries_List_foldl___at___00List_sigmaTR_spec__1___redArg(v_l_u2082_2242_, v___x_2243_, v_l_u2081_2241_);
v___x_2245_ = lean_array_to_list(v___x_2244_);
return v___x_2245_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_sigmaTR(lean_object* v_00_u03b1_2246_, lean_object* v_00_u03c3_2247_, lean_object* v_l_u2081_2248_, lean_object* v_l_u2082_2249_){
_start:
{
lean_object* v___x_2250_; 
v___x_2250_ = lp_batteries_List_sigmaTR___redArg(v_l_u2081_2248_, v_l_u2082_2249_);
return v___x_2250_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sigmaTR_spec__0(lean_object* v_00_u03b1_2251_, lean_object* v_00_u03c3_2252_, lean_object* v_a_2253_, lean_object* v_x_2254_, lean_object* v_x_2255_){
_start:
{
lean_object* v___x_2256_; 
v___x_2256_ = lp_batteries_List_foldl___at___00List_sigmaTR_spec__0___redArg(v_a_2253_, v_x_2254_, v_x_2255_);
return v___x_2256_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_sigmaTR_spec__1(lean_object* v_00_u03b1_2257_, lean_object* v_00_u03c3_2258_, lean_object* v_l_u2082_2259_, lean_object* v_x_2260_, lean_object* v_x_2261_){
_start:
{
lean_object* v___x_2262_; 
v___x_2262_ = lp_batteries_List_foldl___at___00List_sigmaTR_spec__1___redArg(v_l_u2082_2259_, v_x_2260_, v_x_2261_);
return v___x_2262_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_ofFnNthVal___redArg(lean_object* v_n_2263_, lean_object* v_f_2264_, lean_object* v_i_2265_){
_start:
{
uint8_t v___x_2266_; 
v___x_2266_ = lean_nat_dec_lt(v_i_2265_, v_n_2263_);
if (v___x_2266_ == 0)
{
lean_object* v___x_2267_; 
lean_dec(v_i_2265_);
lean_dec(v_f_2264_);
v___x_2267_ = lean_box(0);
return v___x_2267_;
}
else
{
lean_object* v___x_2268_; lean_object* v___x_2269_; 
v___x_2268_ = lean_apply_1(v_f_2264_, v_i_2265_);
v___x_2269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2269_, 0, v___x_2268_);
return v___x_2269_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_ofFnNthVal___redArg___boxed(lean_object* v_n_2270_, lean_object* v_f_2271_, lean_object* v_i_2272_){
_start:
{
lean_object* v_res_2273_; 
v_res_2273_ = lp_batteries_List_ofFnNthVal___redArg(v_n_2270_, v_f_2271_, v_i_2272_);
lean_dec(v_n_2270_);
return v_res_2273_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_ofFnNthVal(lean_object* v_00_u03b1_2274_, lean_object* v_n_2275_, lean_object* v_f_2276_, lean_object* v_i_2277_){
_start:
{
lean_object* v___x_2278_; 
v___x_2278_ = lp_batteries_List_ofFnNthVal___redArg(v_n_2275_, v_f_2276_, v_i_2277_);
return v___x_2278_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_ofFnNthVal___boxed(lean_object* v_00_u03b1_2279_, lean_object* v_n_2280_, lean_object* v_f_2281_, lean_object* v_i_2282_){
_start:
{
lean_object* v_res_2283_; 
v_res_2283_ = lp_batteries_List_ofFnNthVal(v_00_u03b1_2279_, v_n_2280_, v_f_2281_, v_i_2282_);
lean_dec(v_n_2280_);
return v_res_2283_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082___redArg(lean_object* v_R_2286_, lean_object* v_x_2287_, lean_object* v_x_2288_){
_start:
{
if (lean_obj_tag(v_x_2287_) == 1)
{
if (lean_obj_tag(v_x_2288_) == 1)
{
lean_object* v_head_2291_; lean_object* v_tail_2292_; lean_object* v___x_2294_; uint8_t v_isShared_2295_; uint8_t v_isSharedCheck_2321_; 
v_head_2291_ = lean_ctor_get(v_x_2287_, 0);
v_tail_2292_ = lean_ctor_get(v_x_2287_, 1);
v_isSharedCheck_2321_ = !lean_is_exclusive(v_x_2287_);
if (v_isSharedCheck_2321_ == 0)
{
v___x_2294_ = v_x_2287_;
v_isShared_2295_ = v_isSharedCheck_2321_;
goto v_resetjp_2293_;
}
else
{
lean_inc(v_tail_2292_);
lean_inc(v_head_2291_);
lean_dec(v_x_2287_);
v___x_2294_ = lean_box(0);
v_isShared_2295_ = v_isSharedCheck_2321_;
goto v_resetjp_2293_;
}
v_resetjp_2293_:
{
lean_object* v_head_2296_; lean_object* v_tail_2297_; lean_object* v___x_2299_; uint8_t v_isShared_2300_; uint8_t v_isSharedCheck_2320_; 
v_head_2296_ = lean_ctor_get(v_x_2288_, 0);
v_tail_2297_ = lean_ctor_get(v_x_2288_, 1);
v_isSharedCheck_2320_ = !lean_is_exclusive(v_x_2288_);
if (v_isSharedCheck_2320_ == 0)
{
v___x_2299_ = v_x_2288_;
v_isShared_2300_ = v_isSharedCheck_2320_;
goto v_resetjp_2298_;
}
else
{
lean_inc(v_tail_2297_);
lean_inc(v_head_2296_);
lean_dec(v_x_2288_);
v___x_2299_ = lean_box(0);
v_isShared_2300_ = v_isSharedCheck_2320_;
goto v_resetjp_2298_;
}
v_resetjp_2298_:
{
lean_object* v___x_2301_; uint8_t v___x_2302_; 
lean_inc_ref(v_R_2286_);
lean_inc(v_head_2296_);
lean_inc(v_head_2291_);
v___x_2301_ = lean_apply_2(v_R_2286_, v_head_2291_, v_head_2296_);
v___x_2302_ = lean_unbox(v___x_2301_);
if (v___x_2302_ == 0)
{
lean_object* v___x_2303_; 
lean_del_object(v___x_2299_);
lean_dec(v_tail_2297_);
lean_dec(v_head_2296_);
lean_del_object(v___x_2294_);
lean_dec(v_tail_2292_);
lean_dec(v_head_2291_);
lean_dec_ref(v_R_2286_);
v___x_2303_ = ((lean_object*)(lp_batteries_List_takeWhile_u2082___redArg___closed__0));
return v___x_2303_;
}
else
{
lean_object* v___x_2304_; lean_object* v_fst_2305_; lean_object* v_snd_2306_; lean_object* v___x_2308_; uint8_t v_isShared_2309_; uint8_t v_isSharedCheck_2319_; 
v___x_2304_ = lp_batteries_List_takeWhile_u2082___redArg(v_R_2286_, v_tail_2292_, v_tail_2297_);
v_fst_2305_ = lean_ctor_get(v___x_2304_, 0);
v_snd_2306_ = lean_ctor_get(v___x_2304_, 1);
v_isSharedCheck_2319_ = !lean_is_exclusive(v___x_2304_);
if (v_isSharedCheck_2319_ == 0)
{
v___x_2308_ = v___x_2304_;
v_isShared_2309_ = v_isSharedCheck_2319_;
goto v_resetjp_2307_;
}
else
{
lean_inc(v_snd_2306_);
lean_inc(v_fst_2305_);
lean_dec(v___x_2304_);
v___x_2308_ = lean_box(0);
v_isShared_2309_ = v_isSharedCheck_2319_;
goto v_resetjp_2307_;
}
v_resetjp_2307_:
{
lean_object* v___x_2311_; 
if (v_isShared_2300_ == 0)
{
lean_ctor_set(v___x_2299_, 1, v_fst_2305_);
lean_ctor_set(v___x_2299_, 0, v_head_2291_);
v___x_2311_ = v___x_2299_;
goto v_reusejp_2310_;
}
else
{
lean_object* v_reuseFailAlloc_2318_; 
v_reuseFailAlloc_2318_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2318_, 0, v_head_2291_);
lean_ctor_set(v_reuseFailAlloc_2318_, 1, v_fst_2305_);
v___x_2311_ = v_reuseFailAlloc_2318_;
goto v_reusejp_2310_;
}
v_reusejp_2310_:
{
lean_object* v___x_2313_; 
if (v_isShared_2295_ == 0)
{
lean_ctor_set(v___x_2294_, 1, v_snd_2306_);
lean_ctor_set(v___x_2294_, 0, v_head_2296_);
v___x_2313_ = v___x_2294_;
goto v_reusejp_2312_;
}
else
{
lean_object* v_reuseFailAlloc_2317_; 
v_reuseFailAlloc_2317_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2317_, 0, v_head_2296_);
lean_ctor_set(v_reuseFailAlloc_2317_, 1, v_snd_2306_);
v___x_2313_ = v_reuseFailAlloc_2317_;
goto v_reusejp_2312_;
}
v_reusejp_2312_:
{
lean_object* v___x_2315_; 
if (v_isShared_2309_ == 0)
{
lean_ctor_set(v___x_2308_, 1, v___x_2313_);
lean_ctor_set(v___x_2308_, 0, v___x_2311_);
v___x_2315_ = v___x_2308_;
goto v_reusejp_2314_;
}
else
{
lean_object* v_reuseFailAlloc_2316_; 
v_reuseFailAlloc_2316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2316_, 0, v___x_2311_);
lean_ctor_set(v_reuseFailAlloc_2316_, 1, v___x_2313_);
v___x_2315_ = v_reuseFailAlloc_2316_;
goto v_reusejp_2314_;
}
v_reusejp_2314_:
{
return v___x_2315_;
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
lean_dec_ref_known(v_x_2287_, 2);
lean_dec(v_x_2288_);
lean_dec_ref(v_R_2286_);
goto v___jp_2289_;
}
}
else
{
lean_dec(v_x_2288_);
lean_dec(v_x_2287_);
lean_dec_ref(v_R_2286_);
goto v___jp_2289_;
}
v___jp_2289_:
{
lean_object* v___x_2290_; 
v___x_2290_ = ((lean_object*)(lp_batteries_List_takeWhile_u2082___redArg___closed__0));
return v___x_2290_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082(lean_object* v_00_u03b1_2322_, lean_object* v_00_u03b2_2323_, lean_object* v_R_2324_, lean_object* v_x_2325_, lean_object* v_x_2326_){
_start:
{
lean_object* v___x_2327_; 
v___x_2327_ = lp_batteries_List_takeWhile_u2082___redArg(v_R_2324_, v_x_2325_, v_x_2326_);
return v___x_2327_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082TR_go___redArg(lean_object* v_R_2328_, lean_object* v_a_2329_, lean_object* v_a_2330_, lean_object* v_a_2331_, lean_object* v_a_2332_){
_start:
{
lean_object* v_acca_2334_; lean_object* v_accb_2335_; 
if (lean_obj_tag(v_a_2329_) == 1)
{
if (lean_obj_tag(v_a_2330_) == 1)
{
lean_object* v_head_2339_; lean_object* v_tail_2340_; lean_object* v___x_2342_; uint8_t v_isShared_2343_; uint8_t v_isSharedCheck_2362_; 
v_head_2339_ = lean_ctor_get(v_a_2329_, 0);
v_tail_2340_ = lean_ctor_get(v_a_2329_, 1);
v_isSharedCheck_2362_ = !lean_is_exclusive(v_a_2329_);
if (v_isSharedCheck_2362_ == 0)
{
v___x_2342_ = v_a_2329_;
v_isShared_2343_ = v_isSharedCheck_2362_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_tail_2340_);
lean_inc(v_head_2339_);
lean_dec(v_a_2329_);
v___x_2342_ = lean_box(0);
v_isShared_2343_ = v_isSharedCheck_2362_;
goto v_resetjp_2341_;
}
v_resetjp_2341_:
{
lean_object* v_head_2344_; lean_object* v_tail_2345_; lean_object* v___x_2347_; uint8_t v_isShared_2348_; uint8_t v_isSharedCheck_2361_; 
v_head_2344_ = lean_ctor_get(v_a_2330_, 0);
v_tail_2345_ = lean_ctor_get(v_a_2330_, 1);
v_isSharedCheck_2361_ = !lean_is_exclusive(v_a_2330_);
if (v_isSharedCheck_2361_ == 0)
{
v___x_2347_ = v_a_2330_;
v_isShared_2348_ = v_isSharedCheck_2361_;
goto v_resetjp_2346_;
}
else
{
lean_inc(v_tail_2345_);
lean_inc(v_head_2344_);
lean_dec(v_a_2330_);
v___x_2347_ = lean_box(0);
v_isShared_2348_ = v_isSharedCheck_2361_;
goto v_resetjp_2346_;
}
v_resetjp_2346_:
{
lean_object* v___x_2349_; uint8_t v___x_2350_; 
lean_inc_ref(v_R_2328_);
lean_inc(v_head_2344_);
lean_inc(v_head_2339_);
v___x_2349_ = lean_apply_2(v_R_2328_, v_head_2339_, v_head_2344_);
v___x_2350_ = lean_unbox(v___x_2349_);
if (v___x_2350_ == 0)
{
lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___x_2353_; 
lean_del_object(v___x_2347_);
lean_dec(v_tail_2345_);
lean_dec(v_head_2344_);
lean_del_object(v___x_2342_);
lean_dec(v_tail_2340_);
lean_dec(v_head_2339_);
lean_dec_ref(v_R_2328_);
v___x_2351_ = l_List_reverse___redArg(v_a_2331_);
v___x_2352_ = l_List_reverse___redArg(v_a_2332_);
v___x_2353_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2353_, 0, v___x_2351_);
lean_ctor_set(v___x_2353_, 1, v___x_2352_);
return v___x_2353_;
}
else
{
lean_object* v___x_2355_; 
if (v_isShared_2348_ == 0)
{
lean_ctor_set(v___x_2347_, 1, v_a_2331_);
lean_ctor_set(v___x_2347_, 0, v_head_2339_);
v___x_2355_ = v___x_2347_;
goto v_reusejp_2354_;
}
else
{
lean_object* v_reuseFailAlloc_2360_; 
v_reuseFailAlloc_2360_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2360_, 0, v_head_2339_);
lean_ctor_set(v_reuseFailAlloc_2360_, 1, v_a_2331_);
v___x_2355_ = v_reuseFailAlloc_2360_;
goto v_reusejp_2354_;
}
v_reusejp_2354_:
{
lean_object* v___x_2357_; 
if (v_isShared_2343_ == 0)
{
lean_ctor_set(v___x_2342_, 1, v_a_2332_);
lean_ctor_set(v___x_2342_, 0, v_head_2344_);
v___x_2357_ = v___x_2342_;
goto v_reusejp_2356_;
}
else
{
lean_object* v_reuseFailAlloc_2359_; 
v_reuseFailAlloc_2359_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2359_, 0, v_head_2344_);
lean_ctor_set(v_reuseFailAlloc_2359_, 1, v_a_2332_);
v___x_2357_ = v_reuseFailAlloc_2359_;
goto v_reusejp_2356_;
}
v_reusejp_2356_:
{
v_a_2329_ = v_tail_2340_;
v_a_2330_ = v_tail_2345_;
v_a_2331_ = v___x_2355_;
v_a_2332_ = v___x_2357_;
goto _start;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_a_2329_, 2);
lean_dec(v_a_2330_);
lean_dec_ref(v_R_2328_);
v_acca_2334_ = v_a_2331_;
v_accb_2335_ = v_a_2332_;
goto v___jp_2333_;
}
}
else
{
lean_dec(v_a_2330_);
lean_dec(v_a_2329_);
lean_dec_ref(v_R_2328_);
v_acca_2334_ = v_a_2331_;
v_accb_2335_ = v_a_2332_;
goto v___jp_2333_;
}
v___jp_2333_:
{
lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___x_2338_; 
v___x_2336_ = l_List_reverse___redArg(v_acca_2334_);
v___x_2337_ = l_List_reverse___redArg(v_accb_2335_);
v___x_2338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2338_, 0, v___x_2336_);
lean_ctor_set(v___x_2338_, 1, v___x_2337_);
return v___x_2338_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082TR_go(lean_object* v_00_u03b1_2363_, lean_object* v_00_u03b2_2364_, lean_object* v_R_2365_, lean_object* v_a_2366_, lean_object* v_a_2367_, lean_object* v_a_2368_, lean_object* v_a_2369_){
_start:
{
lean_object* v___x_2370_; 
v___x_2370_ = lp_batteries_List_takeWhile_u2082TR_go___redArg(v_R_2365_, v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_);
return v___x_2370_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082TR___redArg(lean_object* v_R_2371_, lean_object* v_as_2372_, lean_object* v_bs_2373_){
_start:
{
lean_object* v___x_2374_; lean_object* v___x_2375_; 
v___x_2374_ = lean_box(0);
v___x_2375_ = lp_batteries_List_takeWhile_u2082TR_go___redArg(v_R_2371_, v_as_2372_, v_bs_2373_, v___x_2374_, v___x_2374_);
return v___x_2375_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeWhile_u2082TR(lean_object* v_00_u03b1_2376_, lean_object* v_00_u03b2_2377_, lean_object* v_R_2378_, lean_object* v_as_2379_, lean_object* v_bs_2380_){
_start:
{
lean_object* v___x_2381_; lean_object* v___x_2382_; 
v___x_2381_ = lean_box(0);
v___x_2382_ = lp_batteries_List_takeWhile_u2082TR_go___redArg(v_R_2378_, v_as_2379_, v_bs_2380_, v___x_2381_, v___x_2381_);
return v___x_2382_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082TR_go_match__1_splitter___redArg(lean_object* v_x_2383_, lean_object* v_x_2384_, lean_object* v_x_2385_, lean_object* v_x_2386_, lean_object* v_h__1_2387_, lean_object* v_h__2_2388_){
_start:
{
if (lean_obj_tag(v_x_2383_) == 1)
{
if (lean_obj_tag(v_x_2384_) == 1)
{
lean_object* v_head_2389_; lean_object* v_tail_2390_; lean_object* v_head_2391_; lean_object* v_tail_2392_; lean_object* v___x_2393_; 
lean_dec(v_h__2_2388_);
v_head_2389_ = lean_ctor_get(v_x_2383_, 0);
lean_inc(v_head_2389_);
v_tail_2390_ = lean_ctor_get(v_x_2383_, 1);
lean_inc(v_tail_2390_);
lean_dec_ref_known(v_x_2383_, 2);
v_head_2391_ = lean_ctor_get(v_x_2384_, 0);
lean_inc(v_head_2391_);
v_tail_2392_ = lean_ctor_get(v_x_2384_, 1);
lean_inc(v_tail_2392_);
lean_dec_ref_known(v_x_2384_, 2);
v___x_2393_ = lean_apply_6(v_h__1_2387_, v_head_2389_, v_tail_2390_, v_head_2391_, v_tail_2392_, v_x_2385_, v_x_2386_);
return v___x_2393_;
}
else
{
lean_object* v___x_2394_; 
lean_dec(v_h__1_2387_);
v___x_2394_ = lean_apply_5(v_h__2_2388_, v_x_2383_, v_x_2384_, v_x_2385_, v_x_2386_, lean_box(0));
return v___x_2394_;
}
}
else
{
lean_object* v___x_2395_; 
lean_dec(v_h__1_2387_);
v___x_2395_ = lean_apply_5(v_h__2_2388_, v_x_2383_, v_x_2384_, v_x_2385_, v_x_2386_, lean_box(0));
return v___x_2395_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082TR_go_match__1_splitter(lean_object* v_00_u03b1_2396_, lean_object* v_00_u03b2_2397_, lean_object* v_motive_2398_, lean_object* v_x_2399_, lean_object* v_x_2400_, lean_object* v_x_2401_, lean_object* v_x_2402_, lean_object* v_h__1_2403_, lean_object* v_h__2_2404_){
_start:
{
if (lean_obj_tag(v_x_2399_) == 1)
{
if (lean_obj_tag(v_x_2400_) == 1)
{
lean_object* v_head_2405_; lean_object* v_tail_2406_; lean_object* v_head_2407_; lean_object* v_tail_2408_; lean_object* v___x_2409_; 
lean_dec(v_h__2_2404_);
v_head_2405_ = lean_ctor_get(v_x_2399_, 0);
lean_inc(v_head_2405_);
v_tail_2406_ = lean_ctor_get(v_x_2399_, 1);
lean_inc(v_tail_2406_);
lean_dec_ref_known(v_x_2399_, 2);
v_head_2407_ = lean_ctor_get(v_x_2400_, 0);
lean_inc(v_head_2407_);
v_tail_2408_ = lean_ctor_get(v_x_2400_, 1);
lean_inc(v_tail_2408_);
lean_dec_ref_known(v_x_2400_, 2);
v___x_2409_ = lean_apply_6(v_h__1_2403_, v_head_2405_, v_tail_2406_, v_head_2407_, v_tail_2408_, v_x_2401_, v_x_2402_);
return v___x_2409_;
}
else
{
lean_object* v___x_2410_; 
lean_dec(v_h__1_2403_);
v___x_2410_ = lean_apply_5(v_h__2_2404_, v_x_2399_, v_x_2400_, v_x_2401_, v_x_2402_, lean_box(0));
return v___x_2410_;
}
}
else
{
lean_object* v___x_2411_; 
lean_dec(v_h__1_2403_);
v___x_2411_ = lean_apply_5(v_h__2_2404_, v_x_2399_, v_x_2400_, v_x_2401_, v_x_2402_, lean_box(0));
return v___x_2411_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082_match__3_splitter___redArg(lean_object* v_x_2412_, lean_object* v_x_2413_, lean_object* v_h__1_2414_, lean_object* v_h__2_2415_){
_start:
{
if (lean_obj_tag(v_x_2412_) == 1)
{
if (lean_obj_tag(v_x_2413_) == 1)
{
lean_object* v_head_2416_; lean_object* v_tail_2417_; lean_object* v_head_2418_; lean_object* v_tail_2419_; lean_object* v___x_2420_; 
lean_dec(v_h__2_2415_);
v_head_2416_ = lean_ctor_get(v_x_2412_, 0);
lean_inc(v_head_2416_);
v_tail_2417_ = lean_ctor_get(v_x_2412_, 1);
lean_inc(v_tail_2417_);
lean_dec_ref_known(v_x_2412_, 2);
v_head_2418_ = lean_ctor_get(v_x_2413_, 0);
lean_inc(v_head_2418_);
v_tail_2419_ = lean_ctor_get(v_x_2413_, 1);
lean_inc(v_tail_2419_);
lean_dec_ref_known(v_x_2413_, 2);
v___x_2420_ = lean_apply_4(v_h__1_2414_, v_head_2416_, v_tail_2417_, v_head_2418_, v_tail_2419_);
return v___x_2420_;
}
else
{
lean_object* v___x_2421_; 
lean_dec(v_h__1_2414_);
v___x_2421_ = lean_apply_3(v_h__2_2415_, v_x_2412_, v_x_2413_, lean_box(0));
return v___x_2421_;
}
}
else
{
lean_object* v___x_2422_; 
lean_dec(v_h__1_2414_);
v___x_2422_ = lean_apply_3(v_h__2_2415_, v_x_2412_, v_x_2413_, lean_box(0));
return v___x_2422_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082_match__3_splitter(lean_object* v_00_u03b1_2423_, lean_object* v_00_u03b2_2424_, lean_object* v_motive_2425_, lean_object* v_x_2426_, lean_object* v_x_2427_, lean_object* v_h__1_2428_, lean_object* v_h__2_2429_){
_start:
{
if (lean_obj_tag(v_x_2426_) == 1)
{
if (lean_obj_tag(v_x_2427_) == 1)
{
lean_object* v_head_2430_; lean_object* v_tail_2431_; lean_object* v_head_2432_; lean_object* v_tail_2433_; lean_object* v___x_2434_; 
lean_dec(v_h__2_2429_);
v_head_2430_ = lean_ctor_get(v_x_2426_, 0);
lean_inc(v_head_2430_);
v_tail_2431_ = lean_ctor_get(v_x_2426_, 1);
lean_inc(v_tail_2431_);
lean_dec_ref_known(v_x_2426_, 2);
v_head_2432_ = lean_ctor_get(v_x_2427_, 0);
lean_inc(v_head_2432_);
v_tail_2433_ = lean_ctor_get(v_x_2427_, 1);
lean_inc(v_tail_2433_);
lean_dec_ref_known(v_x_2427_, 2);
v___x_2434_ = lean_apply_4(v_h__1_2428_, v_head_2430_, v_tail_2431_, v_head_2432_, v_tail_2433_);
return v___x_2434_;
}
else
{
lean_object* v___x_2435_; 
lean_dec(v_h__1_2428_);
v___x_2435_ = lean_apply_3(v_h__2_2429_, v_x_2426_, v_x_2427_, lean_box(0));
return v___x_2435_;
}
}
else
{
lean_object* v___x_2436_; 
lean_dec(v_h__1_2428_);
v___x_2436_ = lean_apply_3(v_h__2_2429_, v_x_2426_, v_x_2427_, lean_box(0));
return v___x_2436_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082_match__1_splitter___redArg(lean_object* v_x_2437_, lean_object* v_h__1_2438_){
_start:
{
lean_object* v_fst_2439_; lean_object* v_snd_2440_; lean_object* v___x_2441_; 
v_fst_2439_ = lean_ctor_get(v_x_2437_, 0);
lean_inc(v_fst_2439_);
v_snd_2440_ = lean_ctor_get(v_x_2437_, 1);
lean_inc(v_snd_2440_);
lean_dec_ref(v_x_2437_);
v___x_2441_ = lean_apply_2(v_h__1_2438_, v_fst_2439_, v_snd_2440_);
return v___x_2441_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeWhile_u2082_match__1_splitter(lean_object* v_00_u03b1_2442_, lean_object* v_00_u03b2_2443_, lean_object* v_motive_2444_, lean_object* v_x_2445_, lean_object* v_h__1_2446_){
_start:
{
lean_object* v_fst_2447_; lean_object* v_snd_2448_; lean_object* v___x_2449_; 
v_fst_2447_ = lean_ctor_get(v_x_2445_, 0);
lean_inc(v_fst_2447_);
v_snd_2448_ = lean_ctor_get(v_x_2445_, 1);
lean_inc(v_snd_2448_);
lean_dec_ref(v_x_2445_);
v___x_2449_ = lean_apply_2(v_h__1_2446_, v_fst_2447_, v_snd_2448_);
return v___x_2449_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_pwFilter___redArg___lam__0(lean_object* v_inst_2450_, lean_object* v_x_2451_, lean_object* v_IH_2452_){
_start:
{
lean_object* v___x_2453_; uint8_t v___x_2454_; 
lean_inc(v_x_2451_);
v___x_2453_ = lean_apply_1(v_inst_2450_, v_x_2451_);
lean_inc(v_IH_2452_);
v___x_2454_ = l_List_decidableBAll___redArg(v___x_2453_, v_IH_2452_);
if (v___x_2454_ == 0)
{
lean_dec(v_x_2451_);
return v_IH_2452_;
}
else
{
lean_object* v___x_2455_; 
v___x_2455_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2455_, 0, v_x_2451_);
lean_ctor_set(v___x_2455_, 1, v_IH_2452_);
return v___x_2455_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_pwFilter___redArg(lean_object* v_inst_2456_, lean_object* v_l_2457_){
_start:
{
lean_object* v___f_2458_; lean_object* v___x_2459_; lean_object* v___x_2460_; 
v___f_2458_ = lean_alloc_closure((void*)(lp_batteries_List_pwFilter___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2458_, 0, v_inst_2456_);
v___x_2459_ = lean_box(0);
v___x_2460_ = l_List_foldrTR___redArg(v___f_2458_, v___x_2459_, v_l_2457_);
return v___x_2460_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_pwFilter(lean_object* v_00_u03b1_2461_, lean_object* v_R_2462_, lean_object* v_inst_2463_, lean_object* v_l_2464_){
_start:
{
lean_object* v___x_2465_; 
v___x_2465_ = lp_batteries_List_pwFilter___redArg(v_inst_2463_, v_l_2464_);
return v___x_2465_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel_go___redArg(lean_object* v_h_2466_, lean_object* v_a_2467_, lean_object* v_l_2468_){
_start:
{
if (lean_obj_tag(v_l_2468_) == 0)
{
uint8_t v___x_2469_; 
lean_dec(v_a_2467_);
lean_dec_ref(v_h_2466_);
v___x_2469_ = 1;
return v___x_2469_;
}
else
{
lean_object* v_head_2470_; lean_object* v_tail_2471_; lean_object* v___x_2472_; uint8_t v___x_2473_; 
v_head_2470_ = lean_ctor_get(v_l_2468_, 0);
lean_inc_n(v_head_2470_, 2);
v_tail_2471_ = lean_ctor_get(v_l_2468_, 1);
lean_inc(v_tail_2471_);
lean_dec_ref_known(v_l_2468_, 2);
lean_inc_ref(v_h_2466_);
v___x_2472_ = lean_apply_2(v_h_2466_, v_a_2467_, v_head_2470_);
v___x_2473_ = lean_unbox(v___x_2472_);
if (v___x_2473_ == 0)
{
uint8_t v___x_2474_; 
lean_dec(v_tail_2471_);
lean_dec(v_head_2470_);
lean_dec_ref(v_h_2466_);
v___x_2474_ = lean_unbox(v___x_2472_);
return v___x_2474_;
}
else
{
v_a_2467_ = v_head_2470_;
v_l_2468_ = v_tail_2471_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableIsChainOfDecidableRel_go___redArg___boxed(lean_object* v_h_2476_, lean_object* v_a_2477_, lean_object* v_l_2478_){
_start:
{
uint8_t v_res_2479_; lean_object* v_r_2480_; 
v_res_2479_ = lp_batteries_List_instDecidableIsChainOfDecidableRel_go___redArg(v_h_2476_, v_a_2477_, v_l_2478_);
v_r_2480_ = lean_box(v_res_2479_);
return v_r_2480_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel_go(lean_object* v_00_u03b1_2481_, lean_object* v_R_2482_, lean_object* v_h_2483_, lean_object* v_a_2484_, lean_object* v_l_2485_){
_start:
{
uint8_t v___x_2486_; 
v___x_2486_ = lp_batteries_List_instDecidableIsChainOfDecidableRel_go___redArg(v_h_2483_, v_a_2484_, v_l_2485_);
return v___x_2486_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableIsChainOfDecidableRel_go___boxed(lean_object* v_00_u03b1_2487_, lean_object* v_R_2488_, lean_object* v_h_2489_, lean_object* v_a_2490_, lean_object* v_l_2491_){
_start:
{
uint8_t v_res_2492_; lean_object* v_r_2493_; 
v_res_2492_ = lp_batteries_List_instDecidableIsChainOfDecidableRel_go(v_00_u03b1_2487_, v_R_2488_, v_h_2489_, v_a_2490_, v_l_2491_);
v_r_2493_ = lean_box(v_res_2492_);
return v_r_2493_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(lean_object* v_h_2494_, lean_object* v_x_2495_){
_start:
{
if (lean_obj_tag(v_x_2495_) == 0)
{
uint8_t v___x_2496_; 
lean_dec_ref(v_h_2494_);
v___x_2496_ = 1;
return v___x_2496_;
}
else
{
lean_object* v_head_2497_; lean_object* v_tail_2498_; uint8_t v___x_2499_; 
v_head_2497_ = lean_ctor_get(v_x_2495_, 0);
lean_inc(v_head_2497_);
v_tail_2498_ = lean_ctor_get(v_x_2495_, 1);
lean_inc(v_tail_2498_);
lean_dec_ref_known(v_x_2495_, 2);
v___x_2499_ = lp_batteries_List_instDecidableIsChainOfDecidableRel_go___redArg(v_h_2494_, v_head_2497_, v_tail_2498_);
return v___x_2499_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg___boxed(lean_object* v_h_2500_, lean_object* v_x_2501_){
_start:
{
uint8_t v_res_2502_; lean_object* v_r_2503_; 
v_res_2502_ = lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(v_h_2500_, v_x_2501_);
v_r_2503_ = lean_box(v_res_2502_);
return v_r_2503_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_instDecidableIsChainOfDecidableRel(lean_object* v_00_u03b1_2504_, lean_object* v_R_2505_, lean_object* v_h_2506_, lean_object* v_x_2507_){
_start:
{
uint8_t v___x_2508_; 
v___x_2508_ = lp_batteries_List_instDecidableIsChainOfDecidableRel___redArg(v_h_2506_, v_x_2507_);
return v___x_2508_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_instDecidableIsChainOfDecidableRel___boxed(lean_object* v_00_u03b1_2509_, lean_object* v_R_2510_, lean_object* v_h_2511_, lean_object* v_x_2512_){
_start:
{
uint8_t v_res_2513_; lean_object* v_r_2514_; 
v_res_2513_ = lp_batteries_List_instDecidableIsChainOfDecidableRel(v_00_u03b1_2509_, v_R_2510_, v_h_2511_, v_x_2512_);
v_r_2514_ = lean_box(v_res_2513_);
return v_r_2514_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_eraseDup___redArg___lam__0(lean_object* v_inst_2515_, lean_object* v_a_2516_, lean_object* v_b_2517_){
_start:
{
lean_object* v___x_2518_; uint8_t v___x_2519_; 
v___x_2518_ = lean_apply_2(v_inst_2515_, v_a_2516_, v_b_2517_);
v___x_2519_ = lean_unbox(v___x_2518_);
if (v___x_2519_ == 0)
{
uint8_t v___x_2520_; 
v___x_2520_ = 1;
return v___x_2520_;
}
else
{
uint8_t v___x_2521_; 
v___x_2521_ = 0;
return v___x_2521_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_eraseDup___redArg___lam__0___boxed(lean_object* v_inst_2522_, lean_object* v_a_2523_, lean_object* v_b_2524_){
_start:
{
uint8_t v_res_2525_; lean_object* v_r_2526_; 
v_res_2525_ = lp_batteries_List_eraseDup___redArg___lam__0(v_inst_2522_, v_a_2523_, v_b_2524_);
v_r_2526_ = lean_box(v_res_2525_);
return v_r_2526_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_eraseDup___redArg(lean_object* v_inst_2527_, lean_object* v_l_2528_){
_start:
{
lean_object* v___f_2529_; lean_object* v___x_2530_; 
v___f_2529_ = lean_alloc_closure((void*)(lp_batteries_List_eraseDup___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_2529_, 0, v_inst_2527_);
v___x_2530_ = lp_batteries_List_pwFilter___redArg(v___f_2529_, v_l_2528_);
return v___x_2530_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_eraseDup(lean_object* v_00_u03b1_2531_, lean_object* v_inst_2532_, lean_object* v_l_2533_){
_start:
{
lean_object* v___f_2534_; lean_object* v___x_2535_; 
v___f_2534_ = lean_alloc_closure((void*)(lp_batteries_List_eraseDup___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_2534_, 0, v_inst_2532_);
v___x_2535_ = lp_batteries_List_pwFilter___redArg(v___f_2534_, v_l_2533_);
return v___x_2535_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_rotate___redArg(lean_object* v_l_2536_, lean_object* v_n_2537_){
_start:
{
lean_object* v___x_2538_; lean_object* v___x_2539_; lean_object* v___x_2540_; lean_object* v_fst_2541_; lean_object* v_snd_2542_; lean_object* v___x_2543_; 
v___x_2538_ = l_List_lengthTR___redArg(v_l_2536_);
v___x_2539_ = lean_nat_mod(v_n_2537_, v___x_2538_);
lean_dec(v___x_2538_);
v___x_2540_ = l_List_splitAt___redArg(v___x_2539_, v_l_2536_);
v_fst_2541_ = lean_ctor_get(v___x_2540_, 0);
lean_inc(v_fst_2541_);
v_snd_2542_ = lean_ctor_get(v___x_2540_, 1);
lean_inc(v_snd_2542_);
lean_dec_ref(v___x_2540_);
v___x_2543_ = l_List_appendTR___redArg(v_snd_2542_, v_fst_2541_);
return v___x_2543_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_rotate___redArg___boxed(lean_object* v_l_2544_, lean_object* v_n_2545_){
_start:
{
lean_object* v_res_2546_; 
v_res_2546_ = lp_batteries_List_rotate___redArg(v_l_2544_, v_n_2545_);
lean_dec(v_n_2545_);
return v_res_2546_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_rotate(lean_object* v_00_u03b1_2547_, lean_object* v_l_2548_, lean_object* v_n_2549_){
_start:
{
lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; lean_object* v_fst_2553_; lean_object* v_snd_2554_; lean_object* v___x_2555_; 
v___x_2550_ = l_List_lengthTR___redArg(v_l_2548_);
v___x_2551_ = lean_nat_mod(v_n_2549_, v___x_2550_);
lean_dec(v___x_2550_);
v___x_2552_ = l_List_splitAt___redArg(v___x_2551_, v_l_2548_);
v_fst_2553_ = lean_ctor_get(v___x_2552_, 0);
lean_inc(v_fst_2553_);
v_snd_2554_ = lean_ctor_get(v___x_2552_, 1);
lean_inc(v_snd_2554_);
lean_dec_ref(v___x_2552_);
v___x_2555_ = l_List_appendTR___redArg(v_snd_2554_, v_fst_2553_);
return v___x_2555_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_rotate___boxed(lean_object* v_00_u03b1_2556_, lean_object* v_l_2557_, lean_object* v_n_2558_){
_start:
{
lean_object* v_res_2559_; 
v_res_2559_ = lp_batteries_List_rotate(v_00_u03b1_2556_, v_l_2557_, v_n_2558_);
lean_dec(v_n_2558_);
return v_res_2559_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_rotate_x27___redArg(lean_object* v_x_2560_, lean_object* v_x_2561_){
_start:
{
if (lean_obj_tag(v_x_2560_) == 0)
{
lean_dec(v_x_2561_);
return v_x_2560_;
}
else
{
lean_object* v_head_2562_; lean_object* v_tail_2563_; lean_object* v_zero_2564_; uint8_t v_isZero_2565_; 
v_head_2562_ = lean_ctor_get(v_x_2560_, 0);
v_tail_2563_ = lean_ctor_get(v_x_2560_, 1);
v_zero_2564_ = lean_unsigned_to_nat(0u);
v_isZero_2565_ = lean_nat_dec_eq(v_x_2561_, v_zero_2564_);
if (v_isZero_2565_ == 1)
{
lean_dec(v_x_2561_);
return v_x_2560_;
}
else
{
lean_object* v___x_2567_; uint8_t v_isShared_2568_; uint8_t v_isSharedCheck_2577_; 
lean_inc(v_tail_2563_);
lean_inc(v_head_2562_);
v_isSharedCheck_2577_ = !lean_is_exclusive(v_x_2560_);
if (v_isSharedCheck_2577_ == 0)
{
lean_object* v_unused_2578_; lean_object* v_unused_2579_; 
v_unused_2578_ = lean_ctor_get(v_x_2560_, 1);
lean_dec(v_unused_2578_);
v_unused_2579_ = lean_ctor_get(v_x_2560_, 0);
lean_dec(v_unused_2579_);
v___x_2567_ = v_x_2560_;
v_isShared_2568_ = v_isSharedCheck_2577_;
goto v_resetjp_2566_;
}
else
{
lean_dec(v_x_2560_);
v___x_2567_ = lean_box(0);
v_isShared_2568_ = v_isSharedCheck_2577_;
goto v_resetjp_2566_;
}
v_resetjp_2566_:
{
lean_object* v_one_2569_; lean_object* v_n_2570_; lean_object* v___x_2571_; lean_object* v___x_2573_; 
v_one_2569_ = lean_unsigned_to_nat(1u);
v_n_2570_ = lean_nat_sub(v_x_2561_, v_one_2569_);
lean_dec(v_x_2561_);
v___x_2571_ = lean_box(0);
if (v_isShared_2568_ == 0)
{
lean_ctor_set(v___x_2567_, 1, v___x_2571_);
v___x_2573_ = v___x_2567_;
goto v_reusejp_2572_;
}
else
{
lean_object* v_reuseFailAlloc_2576_; 
v_reuseFailAlloc_2576_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2576_, 0, v_head_2562_);
lean_ctor_set(v_reuseFailAlloc_2576_, 1, v___x_2571_);
v___x_2573_ = v_reuseFailAlloc_2576_;
goto v_reusejp_2572_;
}
v_reusejp_2572_:
{
lean_object* v___x_2574_; 
v___x_2574_ = l_List_appendTR___redArg(v_tail_2563_, v___x_2573_);
v_x_2560_ = v___x_2574_;
v_x_2561_ = v_n_2570_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_rotate_x27(lean_object* v_00_u03b1_2580_, lean_object* v_x_2581_, lean_object* v_x_2582_){
_start:
{
lean_object* v___x_2583_; 
v___x_2583_ = lp_batteries_List_rotate_x27___redArg(v_x_2581_, v_x_2582_);
return v___x_2583_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_rotate_x27_match__1_splitter___redArg(lean_object* v_x_2584_, lean_object* v_x_2585_, lean_object* v_h__1_2586_, lean_object* v_h__2_2587_, lean_object* v_h__3_2588_){
_start:
{
if (lean_obj_tag(v_x_2584_) == 0)
{
lean_object* v___x_2589_; 
lean_dec(v_h__3_2588_);
lean_dec(v_h__2_2587_);
v___x_2589_ = lean_apply_1(v_h__1_2586_, v_x_2585_);
return v___x_2589_;
}
else
{
lean_object* v_head_2590_; lean_object* v_tail_2591_; lean_object* v_zero_2592_; uint8_t v_isZero_2593_; 
lean_dec(v_h__1_2586_);
v_head_2590_ = lean_ctor_get(v_x_2584_, 0);
v_tail_2591_ = lean_ctor_get(v_x_2584_, 1);
v_zero_2592_ = lean_unsigned_to_nat(0u);
v_isZero_2593_ = lean_nat_dec_eq(v_x_2585_, v_zero_2592_);
if (v_isZero_2593_ == 1)
{
lean_object* v___x_2594_; 
lean_dec(v_h__3_2588_);
lean_dec(v_x_2585_);
v___x_2594_ = lean_apply_2(v_h__2_2587_, v_x_2584_, lean_box(0));
return v___x_2594_;
}
else
{
lean_object* v_one_2595_; lean_object* v_n_2596_; lean_object* v___x_2597_; 
lean_inc(v_tail_2591_);
lean_inc(v_head_2590_);
lean_dec_ref_known(v_x_2584_, 2);
lean_dec(v_h__2_2587_);
v_one_2595_ = lean_unsigned_to_nat(1u);
v_n_2596_ = lean_nat_sub(v_x_2585_, v_one_2595_);
lean_dec(v_x_2585_);
v___x_2597_ = lean_apply_3(v_h__3_2588_, v_head_2590_, v_tail_2591_, v_n_2596_);
return v___x_2597_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_rotate_x27_match__1_splitter(lean_object* v_00_u03b1_2598_, lean_object* v_motive_2599_, lean_object* v_x_2600_, lean_object* v_x_2601_, lean_object* v_h__1_2602_, lean_object* v_h__2_2603_, lean_object* v_h__3_2604_){
_start:
{
if (lean_obj_tag(v_x_2600_) == 0)
{
lean_object* v___x_2605_; 
lean_dec(v_h__3_2604_);
lean_dec(v_h__2_2603_);
v___x_2605_ = lean_apply_1(v_h__1_2602_, v_x_2601_);
return v___x_2605_;
}
else
{
lean_object* v_head_2606_; lean_object* v_tail_2607_; lean_object* v_zero_2608_; uint8_t v_isZero_2609_; 
lean_dec(v_h__1_2602_);
v_head_2606_ = lean_ctor_get(v_x_2600_, 0);
v_tail_2607_ = lean_ctor_get(v_x_2600_, 1);
v_zero_2608_ = lean_unsigned_to_nat(0u);
v_isZero_2609_ = lean_nat_dec_eq(v_x_2601_, v_zero_2608_);
if (v_isZero_2609_ == 1)
{
lean_object* v___x_2610_; 
lean_dec(v_h__3_2604_);
lean_dec(v_x_2601_);
v___x_2610_ = lean_apply_2(v_h__2_2603_, v_x_2600_, lean_box(0));
return v___x_2610_;
}
else
{
lean_object* v_one_2611_; lean_object* v_n_2612_; lean_object* v___x_2613_; 
lean_inc(v_tail_2607_);
lean_inc(v_head_2606_);
lean_dec_ref_known(v_x_2600_, 2);
lean_dec(v_h__2_2603_);
v_one_2611_ = lean_unsigned_to_nat(1u);
v_n_2612_ = lean_nat_sub(v_x_2601_, v_one_2611_);
lean_dec(v_x_2601_);
v___x_2613_ = lean_apply_3(v_h__3_2604_, v_head_2606_, v_tail_2607_, v_n_2612_);
return v___x_2613_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go___redArg___lam__1(lean_object* v_toFunctor_2614_, lean_object* v_f_2615_, lean_object* v_head_2616_, lean_object* v_x1_2617_, lean_object* v_x2_2618_){
_start:
{
lean_object* v_map_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; lean_object* v___x_2622_; 
v_map_2619_ = lean_ctor_get(v_toFunctor_2614_, 0);
lean_inc(v_map_2619_);
lean_dec_ref(v_toFunctor_2614_);
v___x_2620_ = lean_alloc_closure((void*)(l_Array_push___boxed), 3, 2);
lean_closure_set(v___x_2620_, 0, lean_box(0));
lean_closure_set(v___x_2620_, 1, v_x1_2617_);
v___x_2621_ = lean_apply_2(v_f_2615_, v_head_2616_, v_x2_2618_);
v___x_2622_ = lean_apply_4(v_map_2619_, lean_box(0), lean_box(0), v___x_2620_, v___x_2621_);
return v___x_2622_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go___redArg___lam__2(lean_object* v_a_2623_, lean_object* v_inst_2624_, lean_object* v___f_2625_, lean_object* v_tail_2626_, lean_object* v_toBind_2627_, lean_object* v___f_2628_, lean_object* v_b_2629_){
_start:
{
lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; 
v___x_2630_ = lean_array_push(v_a_2623_, v_b_2629_);
v___x_2631_ = l_List_foldlM___redArg(v_inst_2624_, v___f_2625_, v___x_2630_, v_tail_2626_);
v___x_2632_ = lean_apply_4(v_toBind_2627_, lean_box(0), lean_box(0), v___x_2631_, v___f_2628_);
return v___x_2632_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go___redArg(lean_object* v_inst_2633_, lean_object* v_f_2634_, lean_object* v_a_2635_, lean_object* v_a_2636_){
_start:
{
if (lean_obj_tag(v_a_2635_) == 0)
{
lean_object* v_toApplicative_2637_; lean_object* v_toPure_2638_; lean_object* v___x_2639_; lean_object* v___x_2640_; 
v_toApplicative_2637_ = lean_ctor_get(v_inst_2633_, 0);
lean_inc_ref(v_toApplicative_2637_);
lean_dec(v_f_2634_);
lean_dec_ref(v_inst_2633_);
v_toPure_2638_ = lean_ctor_get(v_toApplicative_2637_, 1);
lean_inc(v_toPure_2638_);
lean_dec_ref(v_toApplicative_2637_);
v___x_2639_ = lean_array_to_list(v_a_2636_);
v___x_2640_ = lean_apply_2(v_toPure_2638_, lean_box(0), v___x_2639_);
return v___x_2640_;
}
else
{
lean_object* v_toApplicative_2641_; lean_object* v_toBind_2642_; lean_object* v_toFunctor_2643_; lean_object* v_head_2644_; lean_object* v_tail_2645_; lean_object* v___f_2646_; lean_object* v___f_2647_; lean_object* v___f_2648_; lean_object* v___x_2649_; lean_object* v___x_2650_; 
v_toApplicative_2641_ = lean_ctor_get(v_inst_2633_, 0);
v_toBind_2642_ = lean_ctor_get(v_inst_2633_, 1);
lean_inc_n(v_toBind_2642_, 2);
v_toFunctor_2643_ = lean_ctor_get(v_toApplicative_2641_, 0);
v_head_2644_ = lean_ctor_get(v_a_2635_, 0);
lean_inc_n(v_head_2644_, 3);
v_tail_2645_ = lean_ctor_get(v_a_2635_, 1);
lean_inc_n(v_tail_2645_, 2);
lean_dec_ref_known(v_a_2635_, 2);
lean_inc_n(v_f_2634_, 2);
lean_inc_ref(v_inst_2633_);
v___f_2646_ = lean_alloc_closure((void*)(lp_batteries_List_mapDiagM_go___redArg___lam__0), 4, 3);
lean_closure_set(v___f_2646_, 0, v_inst_2633_);
lean_closure_set(v___f_2646_, 1, v_f_2634_);
lean_closure_set(v___f_2646_, 2, v_tail_2645_);
lean_inc_ref(v_toFunctor_2643_);
v___f_2647_ = lean_alloc_closure((void*)(lp_batteries_List_mapDiagM_go___redArg___lam__1), 5, 3);
lean_closure_set(v___f_2647_, 0, v_toFunctor_2643_);
lean_closure_set(v___f_2647_, 1, v_f_2634_);
lean_closure_set(v___f_2647_, 2, v_head_2644_);
v___f_2648_ = lean_alloc_closure((void*)(lp_batteries_List_mapDiagM_go___redArg___lam__2), 7, 6);
lean_closure_set(v___f_2648_, 0, v_a_2636_);
lean_closure_set(v___f_2648_, 1, v_inst_2633_);
lean_closure_set(v___f_2648_, 2, v___f_2647_);
lean_closure_set(v___f_2648_, 3, v_tail_2645_);
lean_closure_set(v___f_2648_, 4, v_toBind_2642_);
lean_closure_set(v___f_2648_, 5, v___f_2646_);
v___x_2649_ = lean_apply_2(v_f_2634_, v_head_2644_, v_head_2644_);
v___x_2650_ = lean_apply_4(v_toBind_2642_, lean_box(0), lean_box(0), v___x_2649_, v___f_2648_);
return v___x_2650_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go___redArg___lam__0(lean_object* v_inst_2651_, lean_object* v_f_2652_, lean_object* v_tail_2653_, lean_object* v_acc_2654_){
_start:
{
lean_object* v___x_2655_; 
v___x_2655_ = lp_batteries_List_mapDiagM_go___redArg(v_inst_2651_, v_f_2652_, v_tail_2653_, v_acc_2654_);
return v___x_2655_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM_go(lean_object* v_m_2656_, lean_object* v_00_u03b1_2657_, lean_object* v_00_u03b2_2658_, lean_object* v_inst_2659_, lean_object* v_f_2660_, lean_object* v_a_2661_, lean_object* v_a_2662_){
_start:
{
lean_object* v___x_2663_; 
v___x_2663_ = lp_batteries_List_mapDiagM_go___redArg(v_inst_2659_, v_f_2660_, v_a_2661_, v_a_2662_);
return v___x_2663_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM___redArg(lean_object* v_inst_2664_, lean_object* v_f_2665_, lean_object* v_l_2666_){
_start:
{
lean_object* v___x_2667_; lean_object* v___x_2668_; 
v___x_2667_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_2668_ = lp_batteries_List_mapDiagM_go___redArg(v_inst_2664_, v_f_2665_, v_l_2666_, v___x_2667_);
return v___x_2668_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapDiagM(lean_object* v_m_2669_, lean_object* v_00_u03b1_2670_, lean_object* v_00_u03b2_2671_, lean_object* v_inst_2672_, lean_object* v_f_2673_, lean_object* v_l_2674_){
_start:
{
lean_object* v___x_2675_; 
v___x_2675_ = lp_batteries_List_mapDiagM___redArg(v_inst_2672_, v_f_2673_, v_l_2674_);
return v___x_2675_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forDiagM___redArg___lam__1(lean_object* v_f_2676_, lean_object* v_head_2677_, lean_object* v_inst_2678_, lean_object* v_tail_2679_, lean_object* v_toBind_2680_, lean_object* v___f_2681_, lean_object* v_____r_2682_){
_start:
{
lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; 
v___x_2683_ = lean_apply_1(v_f_2676_, v_head_2677_);
v___x_2684_ = l_List_forM___redArg(v_inst_2678_, v_tail_2679_, v___x_2683_);
v___x_2685_ = lean_apply_4(v_toBind_2680_, lean_box(0), lean_box(0), v___x_2684_, v___f_2681_);
return v___x_2685_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forDiagM___redArg(lean_object* v_inst_2686_, lean_object* v_f_2687_, lean_object* v_x_2688_){
_start:
{
if (lean_obj_tag(v_x_2688_) == 0)
{
lean_object* v_toApplicative_2689_; lean_object* v_toPure_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; 
v_toApplicative_2689_ = lean_ctor_get(v_inst_2686_, 0);
lean_inc_ref(v_toApplicative_2689_);
lean_dec(v_f_2687_);
lean_dec_ref(v_inst_2686_);
v_toPure_2690_ = lean_ctor_get(v_toApplicative_2689_, 1);
lean_inc(v_toPure_2690_);
lean_dec_ref(v_toApplicative_2689_);
v___x_2691_ = lean_box(0);
v___x_2692_ = lean_apply_2(v_toPure_2690_, lean_box(0), v___x_2691_);
return v___x_2692_;
}
else
{
lean_object* v_toBind_2693_; lean_object* v_head_2694_; lean_object* v_tail_2695_; lean_object* v___f_2696_; lean_object* v___f_2697_; lean_object* v___x_2698_; lean_object* v___x_2699_; 
v_toBind_2693_ = lean_ctor_get(v_inst_2686_, 1);
lean_inc_n(v_toBind_2693_, 2);
v_head_2694_ = lean_ctor_get(v_x_2688_, 0);
lean_inc_n(v_head_2694_, 3);
v_tail_2695_ = lean_ctor_get(v_x_2688_, 1);
lean_inc_n(v_tail_2695_, 2);
lean_dec_ref_known(v_x_2688_, 2);
lean_inc_n(v_f_2687_, 2);
lean_inc_ref(v_inst_2686_);
v___f_2696_ = lean_alloc_closure((void*)(lp_batteries_List_forDiagM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_2696_, 0, v_inst_2686_);
lean_closure_set(v___f_2696_, 1, v_f_2687_);
lean_closure_set(v___f_2696_, 2, v_tail_2695_);
v___f_2697_ = lean_alloc_closure((void*)(lp_batteries_List_forDiagM___redArg___lam__1), 7, 6);
lean_closure_set(v___f_2697_, 0, v_f_2687_);
lean_closure_set(v___f_2697_, 1, v_head_2694_);
lean_closure_set(v___f_2697_, 2, v_inst_2686_);
lean_closure_set(v___f_2697_, 3, v_tail_2695_);
lean_closure_set(v___f_2697_, 4, v_toBind_2693_);
lean_closure_set(v___f_2697_, 5, v___f_2696_);
v___x_2698_ = lean_apply_2(v_f_2687_, v_head_2694_, v_head_2694_);
v___x_2699_ = lean_apply_4(v_toBind_2693_, lean_box(0), lean_box(0), v___x_2698_, v___f_2697_);
return v___x_2699_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forDiagM___redArg___lam__0(lean_object* v_inst_2700_, lean_object* v_f_2701_, lean_object* v_tail_2702_, lean_object* v_____r_2703_){
_start:
{
lean_object* v___x_2704_; 
v___x_2704_ = lp_batteries_List_forDiagM___redArg(v_inst_2700_, v_f_2701_, v_tail_2702_);
return v___x_2704_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_forDiagM(lean_object* v_m_2705_, lean_object* v_00_u03b1_2706_, lean_object* v_inst_2707_, lean_object* v_f_2708_, lean_object* v_x_2709_){
_start:
{
lean_object* v___x_2710_; 
v___x_2710_ = lp_batteries_List_forDiagM___redArg(v_inst_2707_, v_f_2708_, v_x_2709_);
return v___x_2710_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_getRest___redArg(lean_object* v_inst_2711_, lean_object* v_x_2712_, lean_object* v_x_2713_){
_start:
{
if (lean_obj_tag(v_x_2713_) == 0)
{
lean_object* v___x_2714_; 
lean_dec_ref(v_inst_2711_);
v___x_2714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2714_, 0, v_x_2712_);
return v___x_2714_;
}
else
{
if (lean_obj_tag(v_x_2712_) == 0)
{
lean_object* v___x_2715_; 
lean_dec_ref_known(v_x_2713_, 2);
lean_dec_ref(v_inst_2711_);
v___x_2715_ = lean_box(0);
return v___x_2715_;
}
else
{
lean_object* v_head_2716_; lean_object* v_tail_2717_; lean_object* v_head_2718_; lean_object* v_tail_2719_; lean_object* v___x_2720_; uint8_t v___x_2721_; 
v_head_2716_ = lean_ctor_get(v_x_2713_, 0);
lean_inc(v_head_2716_);
v_tail_2717_ = lean_ctor_get(v_x_2713_, 1);
lean_inc(v_tail_2717_);
lean_dec_ref_known(v_x_2713_, 2);
v_head_2718_ = lean_ctor_get(v_x_2712_, 0);
lean_inc(v_head_2718_);
v_tail_2719_ = lean_ctor_get(v_x_2712_, 1);
lean_inc(v_tail_2719_);
lean_dec_ref_known(v_x_2712_, 2);
lean_inc_ref(v_inst_2711_);
v___x_2720_ = lean_apply_2(v_inst_2711_, v_head_2718_, v_head_2716_);
v___x_2721_ = lean_unbox(v___x_2720_);
if (v___x_2721_ == 0)
{
lean_object* v___x_2722_; 
lean_dec(v_tail_2719_);
lean_dec(v_tail_2717_);
lean_dec_ref(v_inst_2711_);
v___x_2722_ = lean_box(0);
return v___x_2722_;
}
else
{
v_x_2712_ = v_tail_2719_;
v_x_2713_ = v_tail_2717_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_getRest(lean_object* v_00_u03b1_2724_, lean_object* v_inst_2725_, lean_object* v_x_2726_, lean_object* v_x_2727_){
_start:
{
lean_object* v___x_2728_; 
v___x_2728_ = lp_batteries_List_getRest___redArg(v_inst_2725_, v_x_2726_, v_x_2727_);
return v___x_2728_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSlice___redArg(lean_object* v_x_2729_, lean_object* v_x_2730_, lean_object* v_x_2731_){
_start:
{
if (lean_obj_tag(v_x_2731_) == 0)
{
lean_dec(v_x_2730_);
return v_x_2731_;
}
else
{
lean_object* v_head_2732_; lean_object* v_tail_2733_; lean_object* v_zero_2734_; uint8_t v_isZero_2735_; 
v_head_2732_ = lean_ctor_get(v_x_2731_, 0);
v_tail_2733_ = lean_ctor_get(v_x_2731_, 1);
v_zero_2734_ = lean_unsigned_to_nat(0u);
v_isZero_2735_ = lean_nat_dec_eq(v_x_2729_, v_zero_2734_);
if (v_isZero_2735_ == 1)
{
lean_object* v___x_2736_; 
v___x_2736_ = l_List_drop___redArg(v_x_2730_, v_x_2731_);
lean_dec_ref_known(v_x_2731_, 2);
return v___x_2736_;
}
else
{
lean_object* v___x_2738_; uint8_t v_isShared_2739_; uint8_t v_isSharedCheck_2746_; 
lean_inc(v_tail_2733_);
lean_inc(v_head_2732_);
v_isSharedCheck_2746_ = !lean_is_exclusive(v_x_2731_);
if (v_isSharedCheck_2746_ == 0)
{
lean_object* v_unused_2747_; lean_object* v_unused_2748_; 
v_unused_2747_ = lean_ctor_get(v_x_2731_, 1);
lean_dec(v_unused_2747_);
v_unused_2748_ = lean_ctor_get(v_x_2731_, 0);
lean_dec(v_unused_2748_);
v___x_2738_ = v_x_2731_;
v_isShared_2739_ = v_isSharedCheck_2746_;
goto v_resetjp_2737_;
}
else
{
lean_dec(v_x_2731_);
v___x_2738_ = lean_box(0);
v_isShared_2739_ = v_isSharedCheck_2746_;
goto v_resetjp_2737_;
}
v_resetjp_2737_:
{
lean_object* v_one_2740_; lean_object* v_n_2741_; lean_object* v___x_2742_; lean_object* v___x_2744_; 
v_one_2740_ = lean_unsigned_to_nat(1u);
v_n_2741_ = lean_nat_sub(v_x_2729_, v_one_2740_);
v___x_2742_ = lp_batteries_List_dropSlice___redArg(v_n_2741_, v_x_2730_, v_tail_2733_);
lean_dec(v_n_2741_);
if (v_isShared_2739_ == 0)
{
lean_ctor_set(v___x_2738_, 1, v___x_2742_);
v___x_2744_ = v___x_2738_;
goto v_reusejp_2743_;
}
else
{
lean_object* v_reuseFailAlloc_2745_; 
v_reuseFailAlloc_2745_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2745_, 0, v_head_2732_);
lean_ctor_set(v_reuseFailAlloc_2745_, 1, v___x_2742_);
v___x_2744_ = v_reuseFailAlloc_2745_;
goto v_reusejp_2743_;
}
v_reusejp_2743_:
{
return v___x_2744_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSlice___redArg___boxed(lean_object* v_x_2749_, lean_object* v_x_2750_, lean_object* v_x_2751_){
_start:
{
lean_object* v_res_2752_; 
v_res_2752_ = lp_batteries_List_dropSlice___redArg(v_x_2749_, v_x_2750_, v_x_2751_);
lean_dec(v_x_2749_);
return v_res_2752_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSlice(lean_object* v_00_u03b1_2753_, lean_object* v_x_2754_, lean_object* v_x_2755_, lean_object* v_x_2756_){
_start:
{
lean_object* v___x_2757_; 
v___x_2757_ = lp_batteries_List_dropSlice___redArg(v_x_2754_, v_x_2755_, v_x_2756_);
return v___x_2757_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSlice___boxed(lean_object* v_00_u03b1_2758_, lean_object* v_x_2759_, lean_object* v_x_2760_, lean_object* v_x_2761_){
_start:
{
lean_object* v_res_2762_; 
v_res_2762_ = lp_batteries_List_dropSlice(v_00_u03b1_2758_, v_x_2759_, v_x_2760_, v_x_2761_);
lean_dec(v_x_2759_);
return v_res_2762_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSlice_match__1_splitter___redArg(lean_object* v_x_2763_, lean_object* v_x_2764_, lean_object* v_x_2765_, lean_object* v_h__1_2766_, lean_object* v_h__2_2767_, lean_object* v_h__3_2768_){
_start:
{
if (lean_obj_tag(v_x_2765_) == 0)
{
lean_object* v___x_2769_; 
lean_dec(v_h__3_2768_);
lean_dec(v_h__2_2767_);
v___x_2769_ = lean_apply_2(v_h__1_2766_, v_x_2763_, v_x_2764_);
return v___x_2769_;
}
else
{
lean_object* v_head_2770_; lean_object* v_tail_2771_; lean_object* v_zero_2772_; uint8_t v_isZero_2773_; 
lean_dec(v_h__1_2766_);
v_head_2770_ = lean_ctor_get(v_x_2765_, 0);
v_tail_2771_ = lean_ctor_get(v_x_2765_, 1);
v_zero_2772_ = lean_unsigned_to_nat(0u);
v_isZero_2773_ = lean_nat_dec_eq(v_x_2763_, v_zero_2772_);
if (v_isZero_2773_ == 1)
{
lean_object* v___x_2774_; 
lean_dec(v_h__3_2768_);
lean_dec(v_x_2763_);
v___x_2774_ = lean_apply_3(v_h__2_2767_, v_x_2764_, v_x_2765_, lean_box(0));
return v___x_2774_;
}
else
{
lean_object* v_one_2775_; lean_object* v_n_2776_; lean_object* v___x_2777_; 
lean_inc(v_tail_2771_);
lean_inc(v_head_2770_);
lean_dec_ref_known(v_x_2765_, 2);
lean_dec(v_h__2_2767_);
v_one_2775_ = lean_unsigned_to_nat(1u);
v_n_2776_ = lean_nat_sub(v_x_2763_, v_one_2775_);
lean_dec(v_x_2763_);
v___x_2777_ = lean_apply_4(v_h__3_2768_, v_n_2776_, v_x_2764_, v_head_2770_, v_tail_2771_);
return v___x_2777_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSlice_match__1_splitter(lean_object* v_00_u03b1_2778_, lean_object* v_motive_2779_, lean_object* v_x_2780_, lean_object* v_x_2781_, lean_object* v_x_2782_, lean_object* v_h__1_2783_, lean_object* v_h__2_2784_, lean_object* v_h__3_2785_){
_start:
{
if (lean_obj_tag(v_x_2782_) == 0)
{
lean_object* v___x_2786_; 
lean_dec(v_h__3_2785_);
lean_dec(v_h__2_2784_);
v___x_2786_ = lean_apply_2(v_h__1_2783_, v_x_2780_, v_x_2781_);
return v___x_2786_;
}
else
{
lean_object* v_head_2787_; lean_object* v_tail_2788_; lean_object* v_zero_2789_; uint8_t v_isZero_2790_; 
lean_dec(v_h__1_2783_);
v_head_2787_ = lean_ctor_get(v_x_2782_, 0);
v_tail_2788_ = lean_ctor_get(v_x_2782_, 1);
v_zero_2789_ = lean_unsigned_to_nat(0u);
v_isZero_2790_ = lean_nat_dec_eq(v_x_2780_, v_zero_2789_);
if (v_isZero_2790_ == 1)
{
lean_object* v___x_2791_; 
lean_dec(v_h__3_2785_);
lean_dec(v_x_2780_);
v___x_2791_ = lean_apply_3(v_h__2_2784_, v_x_2781_, v_x_2782_, lean_box(0));
return v___x_2791_;
}
else
{
lean_object* v_one_2792_; lean_object* v_n_2793_; lean_object* v___x_2794_; 
lean_inc(v_tail_2788_);
lean_inc(v_head_2787_);
lean_dec_ref_known(v_x_2782_, 2);
lean_dec(v_h__2_2784_);
v_one_2792_ = lean_unsigned_to_nat(1u);
v_n_2793_ = lean_nat_sub(v_x_2780_, v_one_2792_);
lean_dec(v_x_2780_);
v___x_2794_ = lean_apply_4(v_h__3_2785_, v_n_2793_, v_x_2781_, v_head_2787_, v_tail_2788_);
return v___x_2794_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR_go___redArg(lean_object* v_l_2795_, lean_object* v_m_2796_, lean_object* v_a_2797_, lean_object* v_a_2798_, lean_object* v_a_2799_){
_start:
{
if (lean_obj_tag(v_a_2797_) == 0)
{
lean_dec_ref(v_a_2799_);
lean_dec(v_a_2798_);
lean_dec(v_m_2796_);
lean_inc(v_l_2795_);
return v_l_2795_;
}
else
{
lean_object* v_head_2800_; lean_object* v_tail_2801_; lean_object* v_zero_2802_; uint8_t v_isZero_2803_; 
v_head_2800_ = lean_ctor_get(v_a_2797_, 0);
lean_inc(v_head_2800_);
v_tail_2801_ = lean_ctor_get(v_a_2797_, 1);
lean_inc(v_tail_2801_);
lean_dec_ref_known(v_a_2797_, 2);
v_zero_2802_ = lean_unsigned_to_nat(0u);
v_isZero_2803_ = lean_nat_dec_eq(v_a_2798_, v_zero_2802_);
if (v_isZero_2803_ == 1)
{
lean_object* v___x_2804_; lean_object* v___x_2805_; uint8_t v___x_2806_; 
lean_dec(v_head_2800_);
lean_dec(v_a_2798_);
v___x_2804_ = l_List_drop___redArg(v_m_2796_, v_tail_2801_);
lean_dec(v_tail_2801_);
v___x_2805_ = lean_array_get_size(v_a_2799_);
v___x_2806_ = lean_nat_dec_lt(v_zero_2802_, v___x_2805_);
if (v___x_2806_ == 0)
{
lean_dec_ref(v_a_2799_);
return v___x_2804_;
}
else
{
size_t v___x_2807_; size_t v___x_2808_; lean_object* v___x_2809_; 
v___x_2807_ = lean_usize_of_nat(v___x_2805_);
v___x_2808_ = ((size_t)0ULL);
v___x_2809_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_takeDTR_go_spec__0___redArg(v_a_2799_, v___x_2807_, v___x_2808_, v___x_2804_);
lean_dec_ref(v_a_2799_);
return v___x_2809_;
}
}
else
{
lean_object* v_one_2810_; lean_object* v_n_2811_; lean_object* v___x_2812_; 
v_one_2810_ = lean_unsigned_to_nat(1u);
v_n_2811_ = lean_nat_sub(v_a_2798_, v_one_2810_);
lean_dec(v_a_2798_);
v___x_2812_ = lean_array_push(v_a_2799_, v_head_2800_);
v_a_2797_ = v_tail_2801_;
v_a_2798_ = v_n_2811_;
v_a_2799_ = v___x_2812_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR_go___redArg___boxed(lean_object* v_l_2814_, lean_object* v_m_2815_, lean_object* v_a_2816_, lean_object* v_a_2817_, lean_object* v_a_2818_){
_start:
{
lean_object* v_res_2819_; 
v_res_2819_ = lp_batteries_List_dropSliceTR_go___redArg(v_l_2814_, v_m_2815_, v_a_2816_, v_a_2817_, v_a_2818_);
lean_dec(v_l_2814_);
return v_res_2819_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR_go(lean_object* v_00_u03b1_2820_, lean_object* v_l_2821_, lean_object* v_m_2822_, lean_object* v_a_2823_, lean_object* v_a_2824_, lean_object* v_a_2825_){
_start:
{
lean_object* v___x_2826_; 
v___x_2826_ = lp_batteries_List_dropSliceTR_go___redArg(v_l_2821_, v_m_2822_, v_a_2823_, v_a_2824_, v_a_2825_);
return v___x_2826_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR_go___boxed(lean_object* v_00_u03b1_2827_, lean_object* v_l_2828_, lean_object* v_m_2829_, lean_object* v_a_2830_, lean_object* v_a_2831_, lean_object* v_a_2832_){
_start:
{
lean_object* v_res_2833_; 
v_res_2833_ = lp_batteries_List_dropSliceTR_go(v_00_u03b1_2827_, v_l_2828_, v_m_2829_, v_a_2830_, v_a_2831_, v_a_2832_);
lean_dec(v_l_2828_);
return v_res_2833_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR___redArg(lean_object* v_n_2834_, lean_object* v_m_2835_, lean_object* v_l_2836_){
_start:
{
lean_object* v_zero_2837_; uint8_t v_isZero_2838_; 
v_zero_2837_ = lean_unsigned_to_nat(0u);
v_isZero_2838_ = lean_nat_dec_eq(v_m_2835_, v_zero_2837_);
if (v_isZero_2838_ == 1)
{
lean_dec(v_n_2834_);
return v_l_2836_;
}
else
{
lean_object* v_one_2839_; lean_object* v_n_2840_; lean_object* v___x_2841_; lean_object* v___x_2842_; 
v_one_2839_ = lean_unsigned_to_nat(1u);
v_n_2840_ = lean_nat_sub(v_m_2835_, v_one_2839_);
v___x_2841_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_l_2836_);
v___x_2842_ = lp_batteries_List_dropSliceTR_go___redArg(v_l_2836_, v_n_2840_, v_l_2836_, v_n_2834_, v___x_2841_);
lean_dec(v_l_2836_);
return v___x_2842_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR___redArg___boxed(lean_object* v_n_2843_, lean_object* v_m_2844_, lean_object* v_l_2845_){
_start:
{
lean_object* v_res_2846_; 
v_res_2846_ = lp_batteries_List_dropSliceTR___redArg(v_n_2843_, v_m_2844_, v_l_2845_);
lean_dec(v_m_2844_);
return v_res_2846_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR(lean_object* v_00_u03b1_2847_, lean_object* v_n_2848_, lean_object* v_m_2849_, lean_object* v_l_2850_){
_start:
{
lean_object* v_zero_2851_; uint8_t v_isZero_2852_; 
v_zero_2851_ = lean_unsigned_to_nat(0u);
v_isZero_2852_ = lean_nat_dec_eq(v_m_2849_, v_zero_2851_);
if (v_isZero_2852_ == 1)
{
lean_dec(v_n_2848_);
return v_l_2850_;
}
else
{
lean_object* v_one_2853_; lean_object* v_n_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; 
v_one_2853_ = lean_unsigned_to_nat(1u);
v_n_2854_ = lean_nat_sub(v_m_2849_, v_one_2853_);
v___x_2855_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_l_2850_);
v___x_2856_ = lp_batteries_List_dropSliceTR_go___redArg(v_l_2850_, v_n_2854_, v_l_2850_, v_n_2848_, v___x_2855_);
lean_dec(v_l_2850_);
return v___x_2856_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSliceTR___boxed(lean_object* v_00_u03b1_2857_, lean_object* v_n_2858_, lean_object* v_m_2859_, lean_object* v_l_2860_){
_start:
{
lean_object* v_res_2861_; 
v_res_2861_ = lp_batteries_List_dropSliceTR(v_00_u03b1_2857_, v_n_2858_, v_m_2859_, v_l_2860_);
lean_dec(v_m_2859_);
return v_res_2861_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter___redArg(lean_object* v_m_2862_, lean_object* v_h__1_2863_, lean_object* v_h__2_2864_){
_start:
{
lean_object* v_zero_2865_; uint8_t v_isZero_2866_; 
v_zero_2865_ = lean_unsigned_to_nat(0u);
v_isZero_2866_ = lean_nat_dec_eq(v_m_2862_, v_zero_2865_);
if (v_isZero_2866_ == 1)
{
lean_object* v___x_2867_; lean_object* v___x_2868_; 
lean_dec(v_h__2_2864_);
v___x_2867_ = lean_box(0);
v___x_2868_ = lean_apply_1(v_h__1_2863_, v___x_2867_);
return v___x_2868_;
}
else
{
lean_object* v_one_2869_; lean_object* v_n_2870_; lean_object* v___x_2871_; 
lean_dec(v_h__1_2863_);
v_one_2869_ = lean_unsigned_to_nat(1u);
v_n_2870_ = lean_nat_sub(v_m_2862_, v_one_2869_);
v___x_2871_ = lean_apply_1(v_h__2_2864_, v_n_2870_);
return v___x_2871_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter___redArg___boxed(lean_object* v_m_2872_, lean_object* v_h__1_2873_, lean_object* v_h__2_2874_){
_start:
{
lean_object* v_res_2875_; 
v_res_2875_ = lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter___redArg(v_m_2872_, v_h__1_2873_, v_h__2_2874_);
lean_dec(v_m_2872_);
return v_res_2875_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter(lean_object* v_motive_2876_, lean_object* v_m_2877_, lean_object* v_h__1_2878_, lean_object* v_h__2_2879_){
_start:
{
lean_object* v_zero_2880_; uint8_t v_isZero_2881_; 
v_zero_2880_ = lean_unsigned_to_nat(0u);
v_isZero_2881_ = lean_nat_dec_eq(v_m_2877_, v_zero_2880_);
if (v_isZero_2881_ == 1)
{
lean_object* v___x_2882_; lean_object* v___x_2883_; 
lean_dec(v_h__2_2879_);
v___x_2882_ = lean_box(0);
v___x_2883_ = lean_apply_1(v_h__1_2878_, v___x_2882_);
return v___x_2883_;
}
else
{
lean_object* v_one_2884_; lean_object* v_n_2885_; lean_object* v___x_2886_; 
lean_dec(v_h__1_2878_);
v_one_2884_ = lean_unsigned_to_nat(1u);
v_n_2885_ = lean_nat_sub(v_m_2877_, v_one_2884_);
v___x_2886_ = lean_apply_1(v_h__2_2879_, v_n_2885_);
return v___x_2886_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter___boxed(lean_object* v_motive_2887_, lean_object* v_m_2888_, lean_object* v_h__1_2889_, lean_object* v_h__2_2890_){
_start:
{
lean_object* v_res_2891_; 
v_res_2891_ = lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_match__1_splitter(v_motive_2887_, v_m_2888_, v_h__1_2889_, v_h__2_2890_);
lean_dec(v_m_2888_);
return v_res_2891_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_go_match__1_splitter___redArg(lean_object* v_x_2892_, lean_object* v_x_2893_, lean_object* v_x_2894_, lean_object* v_h__1_2895_, lean_object* v_h__2_2896_, lean_object* v_h__3_2897_){
_start:
{
if (lean_obj_tag(v_x_2892_) == 0)
{
lean_object* v___x_2898_; 
lean_dec(v_h__3_2897_);
lean_dec(v_h__2_2896_);
v___x_2898_ = lean_apply_2(v_h__1_2895_, v_x_2893_, v_x_2894_);
return v___x_2898_;
}
else
{
lean_object* v_head_2899_; lean_object* v_tail_2900_; lean_object* v_zero_2901_; uint8_t v_isZero_2902_; 
lean_dec(v_h__1_2895_);
v_head_2899_ = lean_ctor_get(v_x_2892_, 0);
lean_inc(v_head_2899_);
v_tail_2900_ = lean_ctor_get(v_x_2892_, 1);
lean_inc(v_tail_2900_);
lean_dec_ref_known(v_x_2892_, 2);
v_zero_2901_ = lean_unsigned_to_nat(0u);
v_isZero_2902_ = lean_nat_dec_eq(v_x_2893_, v_zero_2901_);
if (v_isZero_2902_ == 1)
{
lean_object* v___x_2903_; 
lean_dec(v_h__3_2897_);
lean_dec(v_x_2893_);
v___x_2903_ = lean_apply_3(v_h__2_2896_, v_head_2899_, v_tail_2900_, v_x_2894_);
return v___x_2903_;
}
else
{
lean_object* v_one_2904_; lean_object* v_n_2905_; lean_object* v___x_2906_; 
lean_dec(v_h__2_2896_);
v_one_2904_ = lean_unsigned_to_nat(1u);
v_n_2905_ = lean_nat_sub(v_x_2893_, v_one_2904_);
lean_dec(v_x_2893_);
v___x_2906_ = lean_apply_4(v_h__3_2897_, v_head_2899_, v_tail_2900_, v_n_2905_, v_x_2894_);
return v___x_2906_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_dropSliceTR_go_match__1_splitter(lean_object* v_00_u03b1_2907_, lean_object* v_motive_2908_, lean_object* v_x_2909_, lean_object* v_x_2910_, lean_object* v_x_2911_, lean_object* v_h__1_2912_, lean_object* v_h__2_2913_, lean_object* v_h__3_2914_){
_start:
{
if (lean_obj_tag(v_x_2909_) == 0)
{
lean_object* v___x_2915_; 
lean_dec(v_h__3_2914_);
lean_dec(v_h__2_2913_);
v___x_2915_ = lean_apply_2(v_h__1_2912_, v_x_2910_, v_x_2911_);
return v___x_2915_;
}
else
{
lean_object* v_head_2916_; lean_object* v_tail_2917_; lean_object* v_zero_2918_; uint8_t v_isZero_2919_; 
lean_dec(v_h__1_2912_);
v_head_2916_ = lean_ctor_get(v_x_2909_, 0);
lean_inc(v_head_2916_);
v_tail_2917_ = lean_ctor_get(v_x_2909_, 1);
lean_inc(v_tail_2917_);
lean_dec_ref_known(v_x_2909_, 2);
v_zero_2918_ = lean_unsigned_to_nat(0u);
v_isZero_2919_ = lean_nat_dec_eq(v_x_2910_, v_zero_2918_);
if (v_isZero_2919_ == 1)
{
lean_object* v___x_2920_; 
lean_dec(v_h__3_2914_);
lean_dec(v_x_2910_);
v___x_2920_ = lean_apply_3(v_h__2_2913_, v_head_2916_, v_tail_2917_, v_x_2911_);
return v___x_2920_;
}
else
{
lean_object* v_one_2921_; lean_object* v_n_2922_; lean_object* v___x_2923_; 
lean_dec(v_h__2_2913_);
v_one_2921_ = lean_unsigned_to_nat(1u);
v_n_2922_ = lean_nat_sub(v_x_2910_, v_one_2921_);
lean_dec(v_x_2910_);
v___x_2923_ = lean_apply_4(v_h__3_2914_, v_head_2916_, v_tail_2917_, v_n_2922_, v_x_2911_);
return v___x_2923_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_zipWithLeft_x27_spec__0___redArg(lean_object* v_f_2924_, lean_object* v_a_2925_, lean_object* v_a_2926_){
_start:
{
if (lean_obj_tag(v_a_2925_) == 0)
{
lean_object* v___x_2927_; 
lean_dec(v_f_2924_);
v___x_2927_ = l_List_reverse___redArg(v_a_2926_);
return v___x_2927_;
}
else
{
lean_object* v_head_2928_; lean_object* v_tail_2929_; lean_object* v___x_2931_; uint8_t v_isShared_2932_; uint8_t v_isSharedCheck_2939_; 
v_head_2928_ = lean_ctor_get(v_a_2925_, 0);
v_tail_2929_ = lean_ctor_get(v_a_2925_, 1);
v_isSharedCheck_2939_ = !lean_is_exclusive(v_a_2925_);
if (v_isSharedCheck_2939_ == 0)
{
v___x_2931_ = v_a_2925_;
v_isShared_2932_ = v_isSharedCheck_2939_;
goto v_resetjp_2930_;
}
else
{
lean_inc(v_tail_2929_);
lean_inc(v_head_2928_);
lean_dec(v_a_2925_);
v___x_2931_ = lean_box(0);
v_isShared_2932_ = v_isSharedCheck_2939_;
goto v_resetjp_2930_;
}
v_resetjp_2930_:
{
lean_object* v___x_2933_; lean_object* v___x_2934_; lean_object* v___x_2936_; 
v___x_2933_ = lean_box(0);
lean_inc(v_f_2924_);
v___x_2934_ = lean_apply_2(v_f_2924_, v_head_2928_, v___x_2933_);
if (v_isShared_2932_ == 0)
{
lean_ctor_set(v___x_2931_, 1, v_a_2926_);
lean_ctor_set(v___x_2931_, 0, v___x_2934_);
v___x_2936_ = v___x_2931_;
goto v_reusejp_2935_;
}
else
{
lean_object* v_reuseFailAlloc_2938_; 
v_reuseFailAlloc_2938_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2938_, 0, v___x_2934_);
lean_ctor_set(v_reuseFailAlloc_2938_, 1, v_a_2926_);
v___x_2936_ = v_reuseFailAlloc_2938_;
goto v_reusejp_2935_;
}
v_reusejp_2935_:
{
v_a_2925_ = v_tail_2929_;
v_a_2926_ = v___x_2936_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27___redArg(lean_object* v_f_2940_, lean_object* v_x_2941_, lean_object* v_x_2942_){
_start:
{
if (lean_obj_tag(v_x_2941_) == 0)
{
lean_object* v___x_2943_; lean_object* v___x_2944_; 
lean_dec(v_f_2940_);
v___x_2943_ = lean_box(0);
v___x_2944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2944_, 0, v___x_2943_);
lean_ctor_set(v___x_2944_, 1, v_x_2942_);
return v___x_2944_;
}
else
{
if (lean_obj_tag(v_x_2942_) == 0)
{
lean_object* v___x_2945_; lean_object* v___x_2946_; lean_object* v___x_2947_; 
v___x_2945_ = lean_box(0);
v___x_2946_ = lp_batteries_List_mapTR_loop___at___00List_zipWithLeft_x27_spec__0___redArg(v_f_2940_, v_x_2941_, v___x_2945_);
v___x_2947_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2947_, 0, v___x_2946_);
lean_ctor_set(v___x_2947_, 1, v_x_2942_);
return v___x_2947_;
}
else
{
lean_object* v_head_2948_; lean_object* v_tail_2949_; lean_object* v_head_2950_; lean_object* v_tail_2951_; lean_object* v___x_2953_; uint8_t v_isShared_2954_; uint8_t v_isSharedCheck_2970_; 
v_head_2948_ = lean_ctor_get(v_x_2941_, 0);
lean_inc(v_head_2948_);
v_tail_2949_ = lean_ctor_get(v_x_2941_, 1);
lean_inc(v_tail_2949_);
lean_dec_ref_known(v_x_2941_, 2);
v_head_2950_ = lean_ctor_get(v_x_2942_, 0);
v_tail_2951_ = lean_ctor_get(v_x_2942_, 1);
v_isSharedCheck_2970_ = !lean_is_exclusive(v_x_2942_);
if (v_isSharedCheck_2970_ == 0)
{
v___x_2953_ = v_x_2942_;
v_isShared_2954_ = v_isSharedCheck_2970_;
goto v_resetjp_2952_;
}
else
{
lean_inc(v_tail_2951_);
lean_inc(v_head_2950_);
lean_dec(v_x_2942_);
v___x_2953_ = lean_box(0);
v_isShared_2954_ = v_isSharedCheck_2970_;
goto v_resetjp_2952_;
}
v_resetjp_2952_:
{
lean_object* v_r_2955_; lean_object* v_fst_2956_; lean_object* v_snd_2957_; lean_object* v___x_2959_; uint8_t v_isShared_2960_; uint8_t v_isSharedCheck_2969_; 
lean_inc(v_f_2940_);
v_r_2955_ = lp_batteries_List_zipWithLeft_x27___redArg(v_f_2940_, v_tail_2949_, v_tail_2951_);
v_fst_2956_ = lean_ctor_get(v_r_2955_, 0);
v_snd_2957_ = lean_ctor_get(v_r_2955_, 1);
v_isSharedCheck_2969_ = !lean_is_exclusive(v_r_2955_);
if (v_isSharedCheck_2969_ == 0)
{
v___x_2959_ = v_r_2955_;
v_isShared_2960_ = v_isSharedCheck_2969_;
goto v_resetjp_2958_;
}
else
{
lean_inc(v_snd_2957_);
lean_inc(v_fst_2956_);
lean_dec(v_r_2955_);
v___x_2959_ = lean_box(0);
v_isShared_2960_ = v_isSharedCheck_2969_;
goto v_resetjp_2958_;
}
v_resetjp_2958_:
{
lean_object* v___x_2961_; lean_object* v___x_2962_; lean_object* v___x_2964_; 
v___x_2961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2961_, 0, v_head_2950_);
v___x_2962_ = lean_apply_2(v_f_2940_, v_head_2948_, v___x_2961_);
if (v_isShared_2954_ == 0)
{
lean_ctor_set(v___x_2953_, 1, v_fst_2956_);
lean_ctor_set(v___x_2953_, 0, v___x_2962_);
v___x_2964_ = v___x_2953_;
goto v_reusejp_2963_;
}
else
{
lean_object* v_reuseFailAlloc_2968_; 
v_reuseFailAlloc_2968_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2968_, 0, v___x_2962_);
lean_ctor_set(v_reuseFailAlloc_2968_, 1, v_fst_2956_);
v___x_2964_ = v_reuseFailAlloc_2968_;
goto v_reusejp_2963_;
}
v_reusejp_2963_:
{
lean_object* v___x_2966_; 
if (v_isShared_2960_ == 0)
{
lean_ctor_set(v___x_2959_, 0, v___x_2964_);
v___x_2966_ = v___x_2959_;
goto v_reusejp_2965_;
}
else
{
lean_object* v_reuseFailAlloc_2967_; 
v_reuseFailAlloc_2967_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2967_, 0, v___x_2964_);
lean_ctor_set(v_reuseFailAlloc_2967_, 1, v_snd_2957_);
v___x_2966_ = v_reuseFailAlloc_2967_;
goto v_reusejp_2965_;
}
v_reusejp_2965_:
{
return v___x_2966_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27(lean_object* v_00_u03b1_2971_, lean_object* v_00_u03b2_2972_, lean_object* v_00_u03b3_2973_, lean_object* v_f_2974_, lean_object* v_x_2975_, lean_object* v_x_2976_){
_start:
{
lean_object* v___x_2977_; 
v___x_2977_ = lp_batteries_List_zipWithLeft_x27___redArg(v_f_2974_, v_x_2975_, v_x_2976_);
return v___x_2977_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00List_zipWithLeft_x27_spec__0(lean_object* v_00_u03b1_2978_, lean_object* v_00_u03b3_2979_, lean_object* v_00_u03b2_2980_, lean_object* v_f_2981_, lean_object* v_a_2982_, lean_object* v_a_2983_){
_start:
{
lean_object* v___x_2984_; 
v___x_2984_ = lp_batteries_List_mapTR_loop___at___00List_zipWithLeft_x27_spec__0___redArg(v_f_2981_, v_a_2982_, v_a_2983_);
return v___x_2984_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_zipWithLeft_x27_match__1_splitter___redArg(lean_object* v_x_2985_, lean_object* v_x_2986_, lean_object* v_h__1_2987_, lean_object* v_h__2_2988_, lean_object* v_h__3_2989_){
_start:
{
if (lean_obj_tag(v_x_2985_) == 0)
{
lean_object* v___x_2990_; 
lean_dec(v_h__3_2989_);
lean_dec(v_h__2_2988_);
v___x_2990_ = lean_apply_1(v_h__1_2987_, v_x_2986_);
return v___x_2990_;
}
else
{
lean_dec(v_h__1_2987_);
if (lean_obj_tag(v_x_2986_) == 0)
{
lean_object* v_head_2991_; lean_object* v_tail_2992_; lean_object* v___x_2993_; 
lean_dec(v_h__3_2989_);
v_head_2991_ = lean_ctor_get(v_x_2985_, 0);
lean_inc(v_head_2991_);
v_tail_2992_ = lean_ctor_get(v_x_2985_, 1);
lean_inc(v_tail_2992_);
lean_dec_ref_known(v_x_2985_, 2);
v___x_2993_ = lean_apply_2(v_h__2_2988_, v_head_2991_, v_tail_2992_);
return v___x_2993_;
}
else
{
lean_object* v_head_2994_; lean_object* v_tail_2995_; lean_object* v_head_2996_; lean_object* v_tail_2997_; lean_object* v___x_2998_; 
lean_dec(v_h__2_2988_);
v_head_2994_ = lean_ctor_get(v_x_2985_, 0);
lean_inc(v_head_2994_);
v_tail_2995_ = lean_ctor_get(v_x_2985_, 1);
lean_inc(v_tail_2995_);
lean_dec_ref_known(v_x_2985_, 2);
v_head_2996_ = lean_ctor_get(v_x_2986_, 0);
lean_inc(v_head_2996_);
v_tail_2997_ = lean_ctor_get(v_x_2986_, 1);
lean_inc(v_tail_2997_);
lean_dec_ref_known(v_x_2986_, 2);
v___x_2998_ = lean_apply_4(v_h__3_2989_, v_head_2994_, v_tail_2995_, v_head_2996_, v_tail_2997_);
return v___x_2998_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_zipWithLeft_x27_match__1_splitter(lean_object* v_00_u03b1_2999_, lean_object* v_00_u03b2_3000_, lean_object* v_motive_3001_, lean_object* v_x_3002_, lean_object* v_x_3003_, lean_object* v_h__1_3004_, lean_object* v_h__2_3005_, lean_object* v_h__3_3006_){
_start:
{
if (lean_obj_tag(v_x_3002_) == 0)
{
lean_object* v___x_3007_; 
lean_dec(v_h__3_3006_);
lean_dec(v_h__2_3005_);
v___x_3007_ = lean_apply_1(v_h__1_3004_, v_x_3003_);
return v___x_3007_;
}
else
{
lean_dec(v_h__1_3004_);
if (lean_obj_tag(v_x_3003_) == 0)
{
lean_object* v_head_3008_; lean_object* v_tail_3009_; lean_object* v___x_3010_; 
lean_dec(v_h__3_3006_);
v_head_3008_ = lean_ctor_get(v_x_3002_, 0);
lean_inc(v_head_3008_);
v_tail_3009_ = lean_ctor_get(v_x_3002_, 1);
lean_inc(v_tail_3009_);
lean_dec_ref_known(v_x_3002_, 2);
v___x_3010_ = lean_apply_2(v_h__2_3005_, v_head_3008_, v_tail_3009_);
return v___x_3010_;
}
else
{
lean_object* v_head_3011_; lean_object* v_tail_3012_; lean_object* v_head_3013_; lean_object* v_tail_3014_; lean_object* v___x_3015_; 
lean_dec(v_h__2_3005_);
v_head_3011_ = lean_ctor_get(v_x_3002_, 0);
lean_inc(v_head_3011_);
v_tail_3012_ = lean_ctor_get(v_x_3002_, 1);
lean_inc(v_tail_3012_);
lean_dec_ref_known(v_x_3002_, 2);
v_head_3013_ = lean_ctor_get(v_x_3003_, 0);
lean_inc(v_head_3013_);
v_tail_3014_ = lean_ctor_get(v_x_3003_, 1);
lean_inc(v_tail_3014_);
lean_dec_ref_known(v_x_3003_, 2);
v___x_3015_ = lean_apply_4(v_h__3_3006_, v_head_3011_, v_tail_3012_, v_head_3013_, v_tail_3014_);
return v___x_3015_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_zipWithLeft_x27TR_go_spec__0___redArg(lean_object* v_f_3016_, lean_object* v_x_3017_, lean_object* v_x_3018_){
_start:
{
if (lean_obj_tag(v_x_3018_) == 0)
{
lean_dec(v_f_3016_);
return v_x_3017_;
}
else
{
lean_object* v_head_3019_; lean_object* v_tail_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; 
v_head_3019_ = lean_ctor_get(v_x_3018_, 0);
lean_inc(v_head_3019_);
v_tail_3020_ = lean_ctor_get(v_x_3018_, 1);
lean_inc(v_tail_3020_);
lean_dec_ref_known(v_x_3018_, 2);
v___x_3021_ = lean_box(0);
lean_inc(v_f_3016_);
v___x_3022_ = lean_apply_2(v_f_3016_, v_head_3019_, v___x_3021_);
v___x_3023_ = lean_array_push(v_x_3017_, v___x_3022_);
v_x_3017_ = v___x_3023_;
v_x_3018_ = v_tail_3020_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27TR_go___redArg(lean_object* v_f_3025_, lean_object* v_a_3026_, lean_object* v_a_3027_, lean_object* v_a_3028_){
_start:
{
if (lean_obj_tag(v_a_3026_) == 0)
{
lean_object* v___x_3029_; lean_object* v___x_3030_; 
lean_dec(v_f_3025_);
v___x_3029_ = lean_array_to_list(v_a_3028_);
v___x_3030_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3030_, 0, v___x_3029_);
lean_ctor_set(v___x_3030_, 1, v_a_3027_);
return v___x_3030_;
}
else
{
if (lean_obj_tag(v_a_3027_) == 0)
{
lean_object* v___x_3031_; lean_object* v___x_3032_; lean_object* v___x_3033_; 
v___x_3031_ = lp_batteries_List_foldl___at___00List_zipWithLeft_x27TR_go_spec__0___redArg(v_f_3025_, v_a_3028_, v_a_3026_);
v___x_3032_ = lean_array_to_list(v___x_3031_);
v___x_3033_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3033_, 0, v___x_3032_);
lean_ctor_set(v___x_3033_, 1, v_a_3027_);
return v___x_3033_;
}
else
{
lean_object* v_head_3034_; lean_object* v_tail_3035_; lean_object* v_head_3036_; lean_object* v_tail_3037_; lean_object* v___x_3038_; lean_object* v___x_3039_; lean_object* v___x_3040_; 
v_head_3034_ = lean_ctor_get(v_a_3026_, 0);
lean_inc(v_head_3034_);
v_tail_3035_ = lean_ctor_get(v_a_3026_, 1);
lean_inc(v_tail_3035_);
lean_dec_ref_known(v_a_3026_, 2);
v_head_3036_ = lean_ctor_get(v_a_3027_, 0);
lean_inc(v_head_3036_);
v_tail_3037_ = lean_ctor_get(v_a_3027_, 1);
lean_inc(v_tail_3037_);
lean_dec_ref_known(v_a_3027_, 2);
v___x_3038_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3038_, 0, v_head_3036_);
lean_inc(v_f_3025_);
v___x_3039_ = lean_apply_2(v_f_3025_, v_head_3034_, v___x_3038_);
v___x_3040_ = lean_array_push(v_a_3028_, v___x_3039_);
v_a_3026_ = v_tail_3035_;
v_a_3027_ = v_tail_3037_;
v_a_3028_ = v___x_3040_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27TR_go(lean_object* v_00_u03b1_3042_, lean_object* v_00_u03b2_3043_, lean_object* v_00_u03b3_3044_, lean_object* v_f_3045_, lean_object* v_a_3046_, lean_object* v_a_3047_, lean_object* v_a_3048_){
_start:
{
lean_object* v___x_3049_; 
v___x_3049_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v_f_3045_, v_a_3046_, v_a_3047_, v_a_3048_);
return v___x_3049_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_foldl___at___00List_zipWithLeft_x27TR_go_spec__0(lean_object* v_00_u03b3_3050_, lean_object* v_00_u03b1_3051_, lean_object* v_00_u03b2_3052_, lean_object* v_f_3053_, lean_object* v_x_3054_, lean_object* v_x_3055_){
_start:
{
lean_object* v___x_3056_; 
v___x_3056_ = lp_batteries_List_foldl___at___00List_zipWithLeft_x27TR_go_spec__0___redArg(v_f_3053_, v_x_3054_, v_x_3055_);
return v___x_3056_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27TR___redArg(lean_object* v_f_3057_, lean_object* v_as_3058_, lean_object* v_bs_3059_){
_start:
{
lean_object* v___x_3060_; lean_object* v___x_3061_; 
v___x_3060_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_3061_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v_f_3057_, v_as_3058_, v_bs_3059_, v___x_3060_);
return v___x_3061_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft_x27TR(lean_object* v_00_u03b1_3062_, lean_object* v_00_u03b2_3063_, lean_object* v_00_u03b3_3064_, lean_object* v_f_3065_, lean_object* v_as_3066_, lean_object* v_bs_3067_){
_start:
{
lean_object* v___x_3068_; lean_object* v___x_3069_; 
v___x_3068_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_3069_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v_f_3065_, v_as_3066_, v_bs_3067_, v___x_3068_);
return v___x_3069_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_zipWithLeft_x27TR_go_match__1_splitter___redArg(lean_object* v_x_3070_, lean_object* v_x_3071_, lean_object* v_x_3072_, lean_object* v_h__1_3073_, lean_object* v_h__2_3074_, lean_object* v_h__3_3075_){
_start:
{
if (lean_obj_tag(v_x_3070_) == 0)
{
lean_object* v___x_3076_; 
lean_dec(v_h__3_3075_);
lean_dec(v_h__2_3074_);
v___x_3076_ = lean_apply_2(v_h__1_3073_, v_x_3071_, v_x_3072_);
return v___x_3076_;
}
else
{
lean_dec(v_h__1_3073_);
if (lean_obj_tag(v_x_3071_) == 0)
{
lean_object* v___x_3077_; 
lean_dec(v_h__3_3075_);
v___x_3077_ = lean_apply_3(v_h__2_3074_, v_x_3070_, v_x_3072_, lean_box(0));
return v___x_3077_;
}
else
{
lean_object* v_head_3078_; lean_object* v_tail_3079_; lean_object* v_head_3080_; lean_object* v_tail_3081_; lean_object* v___x_3082_; 
lean_dec(v_h__2_3074_);
v_head_3078_ = lean_ctor_get(v_x_3070_, 0);
lean_inc(v_head_3078_);
v_tail_3079_ = lean_ctor_get(v_x_3070_, 1);
lean_inc(v_tail_3079_);
lean_dec_ref_known(v_x_3070_, 2);
v_head_3080_ = lean_ctor_get(v_x_3071_, 0);
lean_inc(v_head_3080_);
v_tail_3081_ = lean_ctor_get(v_x_3071_, 1);
lean_inc(v_tail_3081_);
lean_dec_ref_known(v_x_3071_, 2);
v___x_3082_ = lean_apply_5(v_h__3_3075_, v_head_3078_, v_tail_3079_, v_head_3080_, v_tail_3081_, v_x_3072_);
return v___x_3082_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_zipWithLeft_x27TR_go_match__1_splitter(lean_object* v_00_u03b1_3083_, lean_object* v_00_u03b2_3084_, lean_object* v_00_u03b3_3085_, lean_object* v_motive_3086_, lean_object* v_x_3087_, lean_object* v_x_3088_, lean_object* v_x_3089_, lean_object* v_h__1_3090_, lean_object* v_h__2_3091_, lean_object* v_h__3_3092_){
_start:
{
if (lean_obj_tag(v_x_3087_) == 0)
{
lean_object* v___x_3093_; 
lean_dec(v_h__3_3092_);
lean_dec(v_h__2_3091_);
v___x_3093_ = lean_apply_2(v_h__1_3090_, v_x_3088_, v_x_3089_);
return v___x_3093_;
}
else
{
lean_dec(v_h__1_3090_);
if (lean_obj_tag(v_x_3088_) == 0)
{
lean_object* v___x_3094_; 
lean_dec(v_h__3_3092_);
v___x_3094_ = lean_apply_3(v_h__2_3091_, v_x_3087_, v_x_3089_, lean_box(0));
return v___x_3094_;
}
else
{
lean_object* v_head_3095_; lean_object* v_tail_3096_; lean_object* v_head_3097_; lean_object* v_tail_3098_; lean_object* v___x_3099_; 
lean_dec(v_h__2_3091_);
v_head_3095_ = lean_ctor_get(v_x_3087_, 0);
lean_inc(v_head_3095_);
v_tail_3096_ = lean_ctor_get(v_x_3087_, 1);
lean_inc(v_tail_3096_);
lean_dec_ref_known(v_x_3087_, 2);
v_head_3097_ = lean_ctor_get(v_x_3088_, 0);
lean_inc(v_head_3097_);
v_tail_3098_ = lean_ctor_get(v_x_3088_, 1);
lean_inc(v_tail_3098_);
lean_dec_ref_known(v_x_3088_, 2);
v___x_3099_ = lean_apply_5(v_h__3_3092_, v_head_3095_, v_tail_3096_, v_head_3097_, v_tail_3098_, v_x_3089_);
return v___x_3099_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight_x27___redArg(lean_object* v_f_3100_, lean_object* v_as_3101_, lean_object* v_bs_3102_){
_start:
{
lean_object* v___x_3103_; lean_object* v___x_3104_; lean_object* v___x_3105_; 
v___x_3103_ = lean_alloc_closure((void*)(l_flip), 6, 4);
lean_closure_set(v___x_3103_, 0, lean_box(0));
lean_closure_set(v___x_3103_, 1, lean_box(0));
lean_closure_set(v___x_3103_, 2, lean_box(0));
lean_closure_set(v___x_3103_, 3, v_f_3100_);
v___x_3104_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_3105_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v___x_3103_, v_bs_3102_, v_as_3101_, v___x_3104_);
return v___x_3105_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight_x27(lean_object* v_00_u03b1_3106_, lean_object* v_00_u03b2_3107_, lean_object* v_00_u03b3_3108_, lean_object* v_f_3109_, lean_object* v_as_3110_, lean_object* v_bs_3111_){
_start:
{
lean_object* v___x_3112_; lean_object* v___x_3113_; lean_object* v___x_3114_; 
v___x_3112_ = lean_alloc_closure((void*)(l_flip), 6, 4);
lean_closure_set(v___x_3112_, 0, lean_box(0));
lean_closure_set(v___x_3112_, 1, lean_box(0));
lean_closure_set(v___x_3112_, 2, lean_box(0));
lean_closure_set(v___x_3112_, 3, v_f_3109_);
v___x_3113_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_3114_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v___x_3112_, v_bs_3111_, v_as_3110_, v___x_3113_);
return v___x_3114_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft_x27___redArg___lam__0(lean_object* v_fst_3115_, lean_object* v_snd_3116_){
_start:
{
lean_object* v___x_3117_; 
v___x_3117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3117_, 0, v_fst_3115_);
lean_ctor_set(v___x_3117_, 1, v_snd_3116_);
return v___x_3117_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft_x27___redArg(lean_object* v_a_3121_, lean_object* v_a_3122_){
_start:
{
lean_object* v___f_3123_; lean_object* v___x_3124_; lean_object* v___x_3125_; 
v___f_3123_ = ((lean_object*)(lp_batteries_List_zipLeft_x27___redArg___closed__0));
v___x_3124_ = ((lean_object*)(lp_batteries_List_zipLeft_x27___redArg___closed__1));
v___x_3125_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v___f_3123_, v_a_3121_, v_a_3122_, v___x_3124_);
return v___x_3125_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft_x27(lean_object* v_00_u03b1_3126_, lean_object* v_00_u03b2_3127_, lean_object* v_a_3128_, lean_object* v_a_3129_){
_start:
{
lean_object* v___f_3130_; lean_object* v___x_3131_; lean_object* v___x_3132_; 
v___f_3130_ = ((lean_object*)(lp_batteries_List_zipLeft_x27___redArg___closed__0));
v___x_3131_ = ((lean_object*)(lp_batteries_List_zipLeft_x27___redArg___closed__1));
v___x_3132_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v___f_3130_, v_a_3128_, v_a_3129_, v___x_3131_);
return v___x_3132_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipRight_x27___redArg___lam__0(lean_object* v_fst_3133_, lean_object* v_snd_3134_){
_start:
{
lean_object* v___x_3135_; 
v___x_3135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3135_, 0, v_fst_3133_);
lean_ctor_set(v___x_3135_, 1, v_snd_3134_);
return v___x_3135_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipRight_x27___redArg(lean_object* v_as_3141_, lean_object* v_bs_3142_){
_start:
{
lean_object* v___x_3143_; lean_object* v___x_3144_; lean_object* v___x_3145_; 
v___x_3143_ = ((lean_object*)(lp_batteries_List_zipRight_x27___redArg___closed__1));
v___x_3144_ = ((lean_object*)(lp_batteries_List_zipRight_x27___redArg___closed__2));
v___x_3145_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v___x_3143_, v_bs_3142_, v_as_3141_, v___x_3144_);
return v___x_3145_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipRight_x27(lean_object* v_00_u03b1_3146_, lean_object* v_00_u03b2_3147_, lean_object* v_as_3148_, lean_object* v_bs_3149_){
_start:
{
lean_object* v___x_3150_; lean_object* v___x_3151_; lean_object* v___x_3152_; 
v___x_3150_ = ((lean_object*)(lp_batteries_List_zipRight_x27___redArg___closed__1));
v___x_3151_ = ((lean_object*)(lp_batteries_List_zipRight_x27___redArg___closed__2));
v___x_3152_ = lp_batteries_List_zipWithLeft_x27TR_go___redArg(v___x_3150_, v_bs_3149_, v_as_3148_, v___x_3151_);
return v___x_3152_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft___redArg(lean_object* v_f_3153_, lean_object* v_x_3154_, lean_object* v_x_3155_){
_start:
{
if (lean_obj_tag(v_x_3154_) == 0)
{
lean_object* v___x_3156_; 
lean_dec(v_x_3155_);
lean_dec(v_f_3153_);
v___x_3156_ = lean_box(0);
return v___x_3156_;
}
else
{
if (lean_obj_tag(v_x_3155_) == 0)
{
lean_object* v___x_3157_; lean_object* v___x_3158_; 
v___x_3157_ = lean_box(0);
v___x_3158_ = lp_batteries_List_mapTR_loop___at___00List_zipWithLeft_x27_spec__0___redArg(v_f_3153_, v_x_3154_, v___x_3157_);
return v___x_3158_;
}
else
{
lean_object* v_head_3159_; lean_object* v_tail_3160_; lean_object* v_head_3161_; lean_object* v_tail_3162_; lean_object* v___x_3164_; uint8_t v_isShared_3165_; uint8_t v_isSharedCheck_3172_; 
v_head_3159_ = lean_ctor_get(v_x_3154_, 0);
lean_inc(v_head_3159_);
v_tail_3160_ = lean_ctor_get(v_x_3154_, 1);
lean_inc(v_tail_3160_);
lean_dec_ref_known(v_x_3154_, 2);
v_head_3161_ = lean_ctor_get(v_x_3155_, 0);
v_tail_3162_ = lean_ctor_get(v_x_3155_, 1);
v_isSharedCheck_3172_ = !lean_is_exclusive(v_x_3155_);
if (v_isSharedCheck_3172_ == 0)
{
v___x_3164_ = v_x_3155_;
v_isShared_3165_ = v_isSharedCheck_3172_;
goto v_resetjp_3163_;
}
else
{
lean_inc(v_tail_3162_);
lean_inc(v_head_3161_);
lean_dec(v_x_3155_);
v___x_3164_ = lean_box(0);
v_isShared_3165_ = v_isSharedCheck_3172_;
goto v_resetjp_3163_;
}
v_resetjp_3163_:
{
lean_object* v___x_3166_; lean_object* v___x_3167_; lean_object* v___x_3168_; lean_object* v___x_3170_; 
v___x_3166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3166_, 0, v_head_3161_);
lean_inc(v_f_3153_);
v___x_3167_ = lean_apply_2(v_f_3153_, v_head_3159_, v___x_3166_);
v___x_3168_ = lp_batteries_List_zipWithLeft___redArg(v_f_3153_, v_tail_3160_, v_tail_3162_);
if (v_isShared_3165_ == 0)
{
lean_ctor_set(v___x_3164_, 1, v___x_3168_);
lean_ctor_set(v___x_3164_, 0, v___x_3167_);
v___x_3170_ = v___x_3164_;
goto v_reusejp_3169_;
}
else
{
lean_object* v_reuseFailAlloc_3171_; 
v_reuseFailAlloc_3171_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3171_, 0, v___x_3167_);
lean_ctor_set(v_reuseFailAlloc_3171_, 1, v___x_3168_);
v___x_3170_ = v_reuseFailAlloc_3171_;
goto v_reusejp_3169_;
}
v_reusejp_3169_:
{
return v___x_3170_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeft(lean_object* v_00_u03b1_3173_, lean_object* v_00_u03b2_3174_, lean_object* v_00_u03b3_3175_, lean_object* v_f_3176_, lean_object* v_x_3177_, lean_object* v_x_3178_){
_start:
{
lean_object* v___x_3179_; 
v___x_3179_ = lp_batteries_List_zipWithLeft___redArg(v_f_3176_, v_x_3177_, v_x_3178_);
return v___x_3179_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR_go___redArg(lean_object* v_f_3180_, lean_object* v_a_3181_, lean_object* v_a_3182_, lean_object* v_a_3183_){
_start:
{
if (lean_obj_tag(v_a_3181_) == 0)
{
lean_object* v___x_3184_; 
lean_dec(v_f_3180_);
v___x_3184_ = lean_array_to_list(v_a_3183_);
return v___x_3184_;
}
else
{
if (lean_obj_tag(v_a_3182_) == 0)
{
lean_object* v___x_3185_; lean_object* v___x_3186_; 
v___x_3185_ = lp_batteries_List_foldl___at___00List_zipWithLeft_x27TR_go_spec__0___redArg(v_f_3180_, v_a_3183_, v_a_3181_);
v___x_3186_ = lean_array_to_list(v___x_3185_);
return v___x_3186_;
}
else
{
lean_object* v_head_3187_; lean_object* v_tail_3188_; lean_object* v_head_3189_; lean_object* v_tail_3190_; lean_object* v___x_3191_; lean_object* v___x_3192_; lean_object* v___x_3193_; 
v_head_3187_ = lean_ctor_get(v_a_3181_, 0);
lean_inc(v_head_3187_);
v_tail_3188_ = lean_ctor_get(v_a_3181_, 1);
lean_inc(v_tail_3188_);
lean_dec_ref_known(v_a_3181_, 2);
v_head_3189_ = lean_ctor_get(v_a_3182_, 0);
v_tail_3190_ = lean_ctor_get(v_a_3182_, 1);
lean_inc(v_head_3189_);
v___x_3191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3191_, 0, v_head_3189_);
lean_inc(v_f_3180_);
v___x_3192_ = lean_apply_2(v_f_3180_, v_head_3187_, v___x_3191_);
v___x_3193_ = lean_array_push(v_a_3183_, v___x_3192_);
v_a_3181_ = v_tail_3188_;
v_a_3182_ = v_tail_3190_;
v_a_3183_ = v___x_3193_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR_go___redArg___boxed(lean_object* v_f_3195_, lean_object* v_a_3196_, lean_object* v_a_3197_, lean_object* v_a_3198_){
_start:
{
lean_object* v_res_3199_; 
v_res_3199_ = lp_batteries_List_zipWithLeftTR_go___redArg(v_f_3195_, v_a_3196_, v_a_3197_, v_a_3198_);
lean_dec(v_a_3197_);
return v_res_3199_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR_go(lean_object* v_00_u03b1_3200_, lean_object* v_00_u03b2_3201_, lean_object* v_00_u03b3_3202_, lean_object* v_f_3203_, lean_object* v_a_3204_, lean_object* v_a_3205_, lean_object* v_a_3206_){
_start:
{
lean_object* v___x_3207_; 
v___x_3207_ = lp_batteries_List_zipWithLeftTR_go___redArg(v_f_3203_, v_a_3204_, v_a_3205_, v_a_3206_);
return v___x_3207_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR_go___boxed(lean_object* v_00_u03b1_3208_, lean_object* v_00_u03b2_3209_, lean_object* v_00_u03b3_3210_, lean_object* v_f_3211_, lean_object* v_a_3212_, lean_object* v_a_3213_, lean_object* v_a_3214_){
_start:
{
lean_object* v_res_3215_; 
v_res_3215_ = lp_batteries_List_zipWithLeftTR_go(v_00_u03b1_3208_, v_00_u03b2_3209_, v_00_u03b3_3210_, v_f_3211_, v_a_3212_, v_a_3213_, v_a_3214_);
lean_dec(v_a_3213_);
return v_res_3215_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR___redArg(lean_object* v_f_3216_, lean_object* v_as_3217_, lean_object* v_bs_3218_){
_start:
{
lean_object* v___x_3219_; lean_object* v___x_3220_; 
v___x_3219_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_3220_ = lp_batteries_List_zipWithLeftTR_go___redArg(v_f_3216_, v_as_3217_, v_bs_3218_, v___x_3219_);
return v___x_3220_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR___redArg___boxed(lean_object* v_f_3221_, lean_object* v_as_3222_, lean_object* v_bs_3223_){
_start:
{
lean_object* v_res_3224_; 
v_res_3224_ = lp_batteries_List_zipWithLeftTR___redArg(v_f_3221_, v_as_3222_, v_bs_3223_);
lean_dec(v_bs_3223_);
return v_res_3224_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR(lean_object* v_00_u03b1_3225_, lean_object* v_00_u03b2_3226_, lean_object* v_00_u03b3_3227_, lean_object* v_f_3228_, lean_object* v_as_3229_, lean_object* v_bs_3230_){
_start:
{
lean_object* v___x_3231_; lean_object* v___x_3232_; 
v___x_3231_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_3232_ = lp_batteries_List_zipWithLeftTR_go___redArg(v_f_3228_, v_as_3229_, v_bs_3230_, v___x_3231_);
return v___x_3232_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithLeftTR___boxed(lean_object* v_00_u03b1_3233_, lean_object* v_00_u03b2_3234_, lean_object* v_00_u03b3_3235_, lean_object* v_f_3236_, lean_object* v_as_3237_, lean_object* v_bs_3238_){
_start:
{
lean_object* v_res_3239_; 
v_res_3239_ = lp_batteries_List_zipWithLeftTR(v_00_u03b1_3233_, v_00_u03b2_3234_, v_00_u03b3_3235_, v_f_3236_, v_as_3237_, v_bs_3238_);
lean_dec(v_bs_3238_);
return v_res_3239_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight___redArg(lean_object* v_f_3240_, lean_object* v_as_3241_, lean_object* v_bs_3242_){
_start:
{
lean_object* v___x_3243_; lean_object* v___x_3244_; lean_object* v___x_3245_; 
v___x_3243_ = lean_alloc_closure((void*)(l_flip), 6, 4);
lean_closure_set(v___x_3243_, 0, lean_box(0));
lean_closure_set(v___x_3243_, 1, lean_box(0));
lean_closure_set(v___x_3243_, 2, lean_box(0));
lean_closure_set(v___x_3243_, 3, v_f_3240_);
v___x_3244_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_3245_ = lp_batteries_List_zipWithLeftTR_go___redArg(v___x_3243_, v_bs_3242_, v_as_3241_, v___x_3244_);
return v___x_3245_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight___redArg___boxed(lean_object* v_f_3246_, lean_object* v_as_3247_, lean_object* v_bs_3248_){
_start:
{
lean_object* v_res_3249_; 
v_res_3249_ = lp_batteries_List_zipWithRight___redArg(v_f_3246_, v_as_3247_, v_bs_3248_);
lean_dec(v_as_3247_);
return v_res_3249_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight(lean_object* v_00_u03b1_3250_, lean_object* v_00_u03b2_3251_, lean_object* v_00_u03b3_3252_, lean_object* v_f_3253_, lean_object* v_as_3254_, lean_object* v_bs_3255_){
_start:
{
lean_object* v___x_3256_; lean_object* v___x_3257_; lean_object* v___x_3258_; 
v___x_3256_ = lean_alloc_closure((void*)(l_flip), 6, 4);
lean_closure_set(v___x_3256_, 0, lean_box(0));
lean_closure_set(v___x_3256_, 1, lean_box(0));
lean_closure_set(v___x_3256_, 2, lean_box(0));
lean_closure_set(v___x_3256_, 3, v_f_3253_);
v___x_3257_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
v___x_3258_ = lp_batteries_List_zipWithLeftTR_go___redArg(v___x_3256_, v_bs_3255_, v_as_3254_, v___x_3257_);
return v___x_3258_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWithRight___boxed(lean_object* v_00_u03b1_3259_, lean_object* v_00_u03b2_3260_, lean_object* v_00_u03b3_3261_, lean_object* v_f_3262_, lean_object* v_as_3263_, lean_object* v_bs_3264_){
_start:
{
lean_object* v_res_3265_; 
v_res_3265_ = lp_batteries_List_zipWithRight(v_00_u03b1_3259_, v_00_u03b2_3260_, v_00_u03b3_3261_, v_f_3262_, v_as_3263_, v_bs_3264_);
lean_dec(v_as_3263_);
return v_res_3265_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft___redArg(lean_object* v_a_3266_, lean_object* v_a_3267_){
_start:
{
lean_object* v___f_3268_; lean_object* v___x_3269_; lean_object* v___x_3270_; 
v___f_3268_ = ((lean_object*)(lp_batteries_List_zipLeft_x27___redArg___closed__0));
v___x_3269_ = ((lean_object*)(lp_batteries_List_zipLeft_x27___redArg___closed__1));
v___x_3270_ = lp_batteries_List_zipWithLeftTR_go___redArg(v___f_3268_, v_a_3266_, v_a_3267_, v___x_3269_);
return v___x_3270_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft___redArg___boxed(lean_object* v_a_3271_, lean_object* v_a_3272_){
_start:
{
lean_object* v_res_3273_; 
v_res_3273_ = lp_batteries_List_zipLeft___redArg(v_a_3271_, v_a_3272_);
lean_dec(v_a_3272_);
return v_res_3273_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft(lean_object* v_00_u03b1_3274_, lean_object* v_00_u03b2_3275_, lean_object* v_a_3276_, lean_object* v_a_3277_){
_start:
{
lean_object* v___f_3278_; lean_object* v___x_3279_; lean_object* v___x_3280_; 
v___f_3278_ = ((lean_object*)(lp_batteries_List_zipLeft_x27___redArg___closed__0));
v___x_3279_ = ((lean_object*)(lp_batteries_List_zipLeft_x27___redArg___closed__1));
v___x_3280_ = lp_batteries_List_zipWithLeftTR_go___redArg(v___f_3278_, v_a_3276_, v_a_3277_, v___x_3279_);
return v___x_3280_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipLeft___boxed(lean_object* v_00_u03b1_3281_, lean_object* v_00_u03b2_3282_, lean_object* v_a_3283_, lean_object* v_a_3284_){
_start:
{
lean_object* v_res_3285_; 
v_res_3285_ = lp_batteries_List_zipLeft(v_00_u03b1_3281_, v_00_u03b2_3282_, v_a_3283_, v_a_3284_);
lean_dec(v_a_3284_);
return v_res_3285_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipRight___redArg(lean_object* v_as_3286_, lean_object* v_bs_3287_){
_start:
{
lean_object* v___x_3288_; lean_object* v___x_3289_; lean_object* v___x_3290_; 
v___x_3288_ = ((lean_object*)(lp_batteries_List_zipRight_x27___redArg___closed__1));
v___x_3289_ = ((lean_object*)(lp_batteries_List_zipRight_x27___redArg___closed__2));
v___x_3290_ = lp_batteries_List_zipWithLeftTR_go___redArg(v___x_3288_, v_bs_3287_, v_as_3286_, v___x_3289_);
return v___x_3290_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipRight___redArg___boxed(lean_object* v_as_3291_, lean_object* v_bs_3292_){
_start:
{
lean_object* v_res_3293_; 
v_res_3293_ = lp_batteries_List_zipRight___redArg(v_as_3291_, v_bs_3292_);
lean_dec(v_as_3291_);
return v_res_3293_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipRight(lean_object* v_00_u03b1_3294_, lean_object* v_00_u03b2_3295_, lean_object* v_as_3296_, lean_object* v_bs_3297_){
_start:
{
lean_object* v___x_3298_; lean_object* v___x_3299_; lean_object* v___x_3300_; 
v___x_3298_ = ((lean_object*)(lp_batteries_List_zipRight_x27___redArg___closed__1));
v___x_3299_ = ((lean_object*)(lp_batteries_List_zipRight_x27___redArg___closed__2));
v___x_3300_ = lp_batteries_List_zipWithLeftTR_go___redArg(v___x_3298_, v_bs_3297_, v_as_3296_, v___x_3299_);
return v___x_3300_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipRight___boxed(lean_object* v_00_u03b1_3301_, lean_object* v_00_u03b2_3302_, lean_object* v_as_3303_, lean_object* v_bs_3304_){
_start:
{
lean_object* v_res_3305_; 
v_res_3305_ = lp_batteries_List_zipRight(v_00_u03b1_3301_, v_00_u03b2_3302_, v_as_3303_, v_bs_3304_);
lean_dec(v_as_3303_);
return v_res_3305_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_allSome___redArg(lean_object* v_l_3326_){
_start:
{
lean_object* v___x_3327_; lean_object* v___x_3328_; lean_object* v___x_3329_; lean_object* v___x_3330_; 
v___x_3327_ = ((lean_object*)(lp_batteries_List_allSome___redArg___closed__9));
v___x_3328_ = ((lean_object*)(lp_batteries_List_allSome___redArg___closed__10));
v___x_3329_ = lean_box(0);
v___x_3330_ = l_List_mapM_loop___redArg(v___x_3327_, v___x_3328_, v_l_3326_, v___x_3329_);
return v___x_3330_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_allSome(lean_object* v_00_u03b1_3331_, lean_object* v_l_3332_){
_start:
{
lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3335_; lean_object* v___x_3336_; 
v___x_3333_ = ((lean_object*)(lp_batteries_List_allSome___redArg___closed__9));
v___x_3334_ = ((lean_object*)(lp_batteries_List_allSome___redArg___closed__10));
v___x_3335_ = lean_box(0);
v___x_3336_ = l_List_mapM_loop___redArg(v___x_3333_, v___x_3334_, v_l_3332_, v___x_3335_);
return v___x_3336_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeList___redArg(lean_object* v_x_3337_, lean_object* v_x_3338_){
_start:
{
if (lean_obj_tag(v_x_3338_) == 0)
{
lean_object* v___x_3339_; lean_object* v___x_3340_; 
v___x_3339_ = lean_box(0);
v___x_3340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3340_, 0, v___x_3339_);
lean_ctor_set(v___x_3340_, 1, v_x_3337_);
return v___x_3340_;
}
else
{
lean_object* v_head_3341_; lean_object* v_tail_3342_; lean_object* v___x_3344_; uint8_t v_isShared_3345_; uint8_t v_isSharedCheck_3362_; 
v_head_3341_ = lean_ctor_get(v_x_3338_, 0);
v_tail_3342_ = lean_ctor_get(v_x_3338_, 1);
v_isSharedCheck_3362_ = !lean_is_exclusive(v_x_3338_);
if (v_isSharedCheck_3362_ == 0)
{
v___x_3344_ = v_x_3338_;
v_isShared_3345_ = v_isSharedCheck_3362_;
goto v_resetjp_3343_;
}
else
{
lean_inc(v_tail_3342_);
lean_inc(v_head_3341_);
lean_dec(v_x_3338_);
v___x_3344_ = lean_box(0);
v_isShared_3345_ = v_isSharedCheck_3362_;
goto v_resetjp_3343_;
}
v_resetjp_3343_:
{
lean_object* v___x_3346_; lean_object* v_fst_3347_; lean_object* v_snd_3348_; lean_object* v___x_3349_; lean_object* v_fst_3350_; lean_object* v_snd_3351_; lean_object* v___x_3353_; uint8_t v_isShared_3354_; uint8_t v_isSharedCheck_3361_; 
v___x_3346_ = l_List_splitAt___redArg(v_head_3341_, v_x_3337_);
v_fst_3347_ = lean_ctor_get(v___x_3346_, 0);
lean_inc(v_fst_3347_);
v_snd_3348_ = lean_ctor_get(v___x_3346_, 1);
lean_inc(v_snd_3348_);
lean_dec_ref(v___x_3346_);
v___x_3349_ = lp_batteries_List_takeList___redArg(v_snd_3348_, v_tail_3342_);
v_fst_3350_ = lean_ctor_get(v___x_3349_, 0);
v_snd_3351_ = lean_ctor_get(v___x_3349_, 1);
v_isSharedCheck_3361_ = !lean_is_exclusive(v___x_3349_);
if (v_isSharedCheck_3361_ == 0)
{
v___x_3353_ = v___x_3349_;
v_isShared_3354_ = v_isSharedCheck_3361_;
goto v_resetjp_3352_;
}
else
{
lean_inc(v_snd_3351_);
lean_inc(v_fst_3350_);
lean_dec(v___x_3349_);
v___x_3353_ = lean_box(0);
v_isShared_3354_ = v_isSharedCheck_3361_;
goto v_resetjp_3352_;
}
v_resetjp_3352_:
{
lean_object* v___x_3356_; 
if (v_isShared_3345_ == 0)
{
lean_ctor_set(v___x_3344_, 1, v_fst_3350_);
lean_ctor_set(v___x_3344_, 0, v_fst_3347_);
v___x_3356_ = v___x_3344_;
goto v_reusejp_3355_;
}
else
{
lean_object* v_reuseFailAlloc_3360_; 
v_reuseFailAlloc_3360_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3360_, 0, v_fst_3347_);
lean_ctor_set(v_reuseFailAlloc_3360_, 1, v_fst_3350_);
v___x_3356_ = v_reuseFailAlloc_3360_;
goto v_reusejp_3355_;
}
v_reusejp_3355_:
{
lean_object* v___x_3358_; 
if (v_isShared_3354_ == 0)
{
lean_ctor_set(v___x_3353_, 0, v___x_3356_);
v___x_3358_ = v___x_3353_;
goto v_reusejp_3357_;
}
else
{
lean_object* v_reuseFailAlloc_3359_; 
v_reuseFailAlloc_3359_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3359_, 0, v___x_3356_);
lean_ctor_set(v_reuseFailAlloc_3359_, 1, v_snd_3351_);
v___x_3358_ = v_reuseFailAlloc_3359_;
goto v_reusejp_3357_;
}
v_reusejp_3357_:
{
return v___x_3358_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeList(lean_object* v_00_u03b1_3363_, lean_object* v_x_3364_, lean_object* v_x_3365_){
_start:
{
lean_object* v___x_3366_; 
v___x_3366_ = lp_batteries_List_takeList___redArg(v_x_3364_, v_x_3365_);
return v___x_3366_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeListTR_go___redArg(lean_object* v_a_3367_, lean_object* v_a_3368_, lean_object* v_a_3369_){
_start:
{
if (lean_obj_tag(v_a_3367_) == 0)
{
lean_object* v___x_3370_; lean_object* v___x_3371_; 
v___x_3370_ = lean_array_to_list(v_a_3369_);
v___x_3371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3371_, 0, v___x_3370_);
lean_ctor_set(v___x_3371_, 1, v_a_3368_);
return v___x_3371_;
}
else
{
lean_object* v_head_3372_; lean_object* v_tail_3373_; lean_object* v___x_3374_; lean_object* v_fst_3375_; lean_object* v_snd_3376_; lean_object* v___x_3377_; 
v_head_3372_ = lean_ctor_get(v_a_3367_, 0);
lean_inc(v_head_3372_);
v_tail_3373_ = lean_ctor_get(v_a_3367_, 1);
lean_inc(v_tail_3373_);
lean_dec_ref_known(v_a_3367_, 2);
v___x_3374_ = l_List_splitAt___redArg(v_head_3372_, v_a_3368_);
v_fst_3375_ = lean_ctor_get(v___x_3374_, 0);
lean_inc(v_fst_3375_);
v_snd_3376_ = lean_ctor_get(v___x_3374_, 1);
lean_inc(v_snd_3376_);
lean_dec_ref(v___x_3374_);
v___x_3377_ = lean_array_push(v_a_3369_, v_fst_3375_);
v_a_3367_ = v_tail_3373_;
v_a_3368_ = v_snd_3376_;
v_a_3369_ = v___x_3377_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeListTR_go(lean_object* v_00_u03b1_3379_, lean_object* v_a_3380_, lean_object* v_a_3381_, lean_object* v_a_3382_){
_start:
{
lean_object* v___x_3383_; 
v___x_3383_ = lp_batteries_List_takeListTR_go___redArg(v_a_3380_, v_a_3381_, v_a_3382_);
return v___x_3383_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeListTR___redArg(lean_object* v_xs_3384_, lean_object* v_ns_3385_){
_start:
{
lean_object* v___x_3386_; lean_object* v___x_3387_; 
v___x_3386_ = ((lean_object*)(lp_batteries_List_tailsTR___redArg___closed__0));
v___x_3387_ = lp_batteries_List_takeListTR_go___redArg(v_ns_3385_, v_xs_3384_, v___x_3386_);
return v___x_3387_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_takeListTR(lean_object* v_00_u03b1_3388_, lean_object* v_xs_3389_, lean_object* v_ns_3390_){
_start:
{
lean_object* v___x_3391_; lean_object* v___x_3392_; 
v___x_3391_ = ((lean_object*)(lp_batteries_List_tailsTR___redArg___closed__0));
v___x_3392_ = lp_batteries_List_takeListTR_go___redArg(v_ns_3390_, v_xs_3389_, v___x_3391_);
return v___x_3392_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeListTR_go_match__1_splitter___redArg(lean_object* v_x_3393_, lean_object* v_x_3394_, lean_object* v_x_3395_, lean_object* v_h__1_3396_, lean_object* v_h__2_3397_){
_start:
{
if (lean_obj_tag(v_x_3393_) == 0)
{
lean_object* v___x_3398_; 
lean_dec(v_h__2_3397_);
v___x_3398_ = lean_apply_2(v_h__1_3396_, v_x_3394_, v_x_3395_);
return v___x_3398_;
}
else
{
lean_object* v_head_3399_; lean_object* v_tail_3400_; lean_object* v___x_3401_; 
lean_dec(v_h__1_3396_);
v_head_3399_ = lean_ctor_get(v_x_3393_, 0);
lean_inc(v_head_3399_);
v_tail_3400_ = lean_ctor_get(v_x_3393_, 1);
lean_inc(v_tail_3400_);
lean_dec_ref_known(v_x_3393_, 2);
v___x_3401_ = lean_apply_4(v_h__2_3397_, v_head_3399_, v_tail_3400_, v_x_3394_, v_x_3395_);
return v___x_3401_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeListTR_go_match__1_splitter(lean_object* v_00_u03b1_3402_, lean_object* v_motive_3403_, lean_object* v_x_3404_, lean_object* v_x_3405_, lean_object* v_x_3406_, lean_object* v_h__1_3407_, lean_object* v_h__2_3408_){
_start:
{
if (lean_obj_tag(v_x_3404_) == 0)
{
lean_object* v___x_3409_; 
lean_dec(v_h__2_3408_);
v___x_3409_ = lean_apply_2(v_h__1_3407_, v_x_3405_, v_x_3406_);
return v___x_3409_;
}
else
{
lean_object* v_head_3410_; lean_object* v_tail_3411_; lean_object* v___x_3412_; 
lean_dec(v_h__1_3407_);
v_head_3410_ = lean_ctor_get(v_x_3404_, 0);
lean_inc(v_head_3410_);
v_tail_3411_ = lean_ctor_get(v_x_3404_, 1);
lean_inc(v_tail_3411_);
lean_dec_ref_known(v_x_3404_, 2);
v___x_3412_ = lean_apply_4(v_h__2_3408_, v_head_3410_, v_tail_3411_, v_x_3405_, v_x_3406_);
return v___x_3412_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_rotate_match__1_splitter___redArg(lean_object* v_x_3413_, lean_object* v_h__1_3414_){
_start:
{
lean_object* v_fst_3415_; lean_object* v_snd_3416_; lean_object* v___x_3417_; 
v_fst_3415_ = lean_ctor_get(v_x_3413_, 0);
lean_inc(v_fst_3415_);
v_snd_3416_ = lean_ctor_get(v_x_3413_, 1);
lean_inc(v_snd_3416_);
lean_dec_ref(v_x_3413_);
v___x_3417_ = lean_apply_2(v_h__1_3414_, v_fst_3415_, v_snd_3416_);
return v___x_3417_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_rotate_match__1_splitter(lean_object* v_00_u03b1_3418_, lean_object* v_motive_3419_, lean_object* v_x_3420_, lean_object* v_h__1_3421_){
_start:
{
lean_object* v_fst_3422_; lean_object* v_snd_3423_; lean_object* v___x_3424_; 
v_fst_3422_ = lean_ctor_get(v_x_3420_, 0);
lean_inc(v_fst_3422_);
v_snd_3423_ = lean_ctor_get(v_x_3420_, 1);
lean_inc(v_snd_3423_);
lean_dec_ref(v_x_3420_);
v___x_3424_ = lean_apply_2(v_h__1_3421_, v_fst_3422_, v_snd_3423_);
return v___x_3424_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeList_match__3_splitter___redArg(lean_object* v_x_3425_, lean_object* v_x_3426_, lean_object* v_h__1_3427_, lean_object* v_h__2_3428_){
_start:
{
if (lean_obj_tag(v_x_3426_) == 0)
{
lean_object* v___x_3429_; 
lean_dec(v_h__2_3428_);
v___x_3429_ = lean_apply_1(v_h__1_3427_, v_x_3425_);
return v___x_3429_;
}
else
{
lean_object* v_head_3430_; lean_object* v_tail_3431_; lean_object* v___x_3432_; 
lean_dec(v_h__1_3427_);
v_head_3430_ = lean_ctor_get(v_x_3426_, 0);
lean_inc(v_head_3430_);
v_tail_3431_ = lean_ctor_get(v_x_3426_, 1);
lean_inc(v_tail_3431_);
lean_dec_ref_known(v_x_3426_, 2);
v___x_3432_ = lean_apply_3(v_h__2_3428_, v_x_3425_, v_head_3430_, v_tail_3431_);
return v___x_3432_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeList_match__3_splitter(lean_object* v_00_u03b1_3433_, lean_object* v_motive_3434_, lean_object* v_x_3435_, lean_object* v_x_3436_, lean_object* v_h__1_3437_, lean_object* v_h__2_3438_){
_start:
{
if (lean_obj_tag(v_x_3436_) == 0)
{
lean_object* v___x_3439_; 
lean_dec(v_h__2_3438_);
v___x_3439_ = lean_apply_1(v_h__1_3437_, v_x_3435_);
return v___x_3439_;
}
else
{
lean_object* v_head_3440_; lean_object* v_tail_3441_; lean_object* v___x_3442_; 
lean_dec(v_h__1_3437_);
v_head_3440_ = lean_ctor_get(v_x_3436_, 0);
lean_inc(v_head_3440_);
v_tail_3441_ = lean_ctor_get(v_x_3436_, 1);
lean_inc(v_tail_3441_);
lean_dec_ref_known(v_x_3436_, 2);
v___x_3442_ = lean_apply_3(v_h__2_3438_, v_x_3435_, v_head_3440_, v_tail_3441_);
return v___x_3442_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeList_match__1_splitter___redArg(lean_object* v_x_3443_, lean_object* v_h__1_3444_){
_start:
{
lean_object* v_fst_3445_; lean_object* v_snd_3446_; lean_object* v___x_3447_; 
v_fst_3445_ = lean_ctor_get(v_x_3443_, 0);
lean_inc(v_fst_3445_);
v_snd_3446_ = lean_ctor_get(v_x_3443_, 1);
lean_inc(v_snd_3446_);
lean_dec_ref(v_x_3443_);
v___x_3447_ = lean_apply_2(v_h__1_3444_, v_fst_3445_, v_snd_3446_);
return v___x_3447_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_takeList_match__1_splitter(lean_object* v_00_u03b1_3448_, lean_object* v_motive_3449_, lean_object* v_x_3450_, lean_object* v_h__1_3451_){
_start:
{
lean_object* v_fst_3452_; lean_object* v_snd_3453_; lean_object* v___x_3454_; 
v_fst_3452_ = lean_ctor_get(v_x_3450_, 0);
lean_inc(v_fst_3452_);
v_snd_3453_ = lean_ctor_get(v_x_3450_, 1);
lean_inc(v_snd_3453_);
lean_dec_ref(v_x_3450_);
v___x_3454_ = lean_apply_2(v_h__1_3451_, v_fst_3452_, v_snd_3453_);
return v___x_3454_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunksAux___redArg(lean_object* v_n_3455_, lean_object* v_x_3456_, lean_object* v_x_3457_){
_start:
{
if (lean_obj_tag(v_x_3456_) == 0)
{
lean_object* v___x_3458_; lean_object* v___x_3459_; 
v___x_3458_ = lean_box(0);
v___x_3459_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3459_, 0, v_x_3456_);
lean_ctor_set(v___x_3459_, 1, v___x_3458_);
return v___x_3459_;
}
else
{
lean_object* v_head_3460_; lean_object* v_tail_3461_; lean_object* v___x_3463_; uint8_t v_isShared_3464_; uint8_t v_isSharedCheck_3497_; 
v_head_3460_ = lean_ctor_get(v_x_3456_, 0);
v_tail_3461_ = lean_ctor_get(v_x_3456_, 1);
v_isSharedCheck_3497_ = !lean_is_exclusive(v_x_3456_);
if (v_isSharedCheck_3497_ == 0)
{
v___x_3463_ = v_x_3456_;
v_isShared_3464_ = v_isSharedCheck_3497_;
goto v_resetjp_3462_;
}
else
{
lean_inc(v_tail_3461_);
lean_inc(v_head_3460_);
lean_dec(v_x_3456_);
v___x_3463_ = lean_box(0);
v_isShared_3464_ = v_isSharedCheck_3497_;
goto v_resetjp_3462_;
}
v_resetjp_3462_:
{
lean_object* v_zero_3465_; uint8_t v_isZero_3466_; 
v_zero_3465_ = lean_unsigned_to_nat(0u);
v_isZero_3466_ = lean_nat_dec_eq(v_x_3457_, v_zero_3465_);
if (v_isZero_3466_ == 1)
{
lean_object* v___x_3467_; lean_object* v_fst_3468_; lean_object* v_snd_3469_; lean_object* v___x_3471_; uint8_t v_isShared_3472_; uint8_t v_isSharedCheck_3481_; 
v___x_3467_ = lp_batteries_List_toChunksAux___redArg(v_n_3455_, v_tail_3461_, v_n_3455_);
v_fst_3468_ = lean_ctor_get(v___x_3467_, 0);
v_snd_3469_ = lean_ctor_get(v___x_3467_, 1);
v_isSharedCheck_3481_ = !lean_is_exclusive(v___x_3467_);
if (v_isSharedCheck_3481_ == 0)
{
v___x_3471_ = v___x_3467_;
v_isShared_3472_ = v_isSharedCheck_3481_;
goto v_resetjp_3470_;
}
else
{
lean_inc(v_snd_3469_);
lean_inc(v_fst_3468_);
lean_dec(v___x_3467_);
v___x_3471_ = lean_box(0);
v_isShared_3472_ = v_isSharedCheck_3481_;
goto v_resetjp_3470_;
}
v_resetjp_3470_:
{
lean_object* v___x_3473_; lean_object* v___x_3475_; 
v___x_3473_ = lean_box(0);
if (v_isShared_3464_ == 0)
{
lean_ctor_set(v___x_3463_, 1, v_fst_3468_);
v___x_3475_ = v___x_3463_;
goto v_reusejp_3474_;
}
else
{
lean_object* v_reuseFailAlloc_3480_; 
v_reuseFailAlloc_3480_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3480_, 0, v_head_3460_);
lean_ctor_set(v_reuseFailAlloc_3480_, 1, v_fst_3468_);
v___x_3475_ = v_reuseFailAlloc_3480_;
goto v_reusejp_3474_;
}
v_reusejp_3474_:
{
lean_object* v___x_3476_; lean_object* v___x_3478_; 
v___x_3476_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3476_, 0, v___x_3475_);
lean_ctor_set(v___x_3476_, 1, v_snd_3469_);
if (v_isShared_3472_ == 0)
{
lean_ctor_set(v___x_3471_, 1, v___x_3476_);
lean_ctor_set(v___x_3471_, 0, v___x_3473_);
v___x_3478_ = v___x_3471_;
goto v_reusejp_3477_;
}
else
{
lean_object* v_reuseFailAlloc_3479_; 
v_reuseFailAlloc_3479_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3479_, 0, v___x_3473_);
lean_ctor_set(v_reuseFailAlloc_3479_, 1, v___x_3476_);
v___x_3478_ = v_reuseFailAlloc_3479_;
goto v_reusejp_3477_;
}
v_reusejp_3477_:
{
return v___x_3478_;
}
}
}
}
else
{
lean_object* v_one_3482_; lean_object* v_n_3483_; lean_object* v___x_3484_; lean_object* v_fst_3485_; lean_object* v_snd_3486_; lean_object* v___x_3488_; uint8_t v_isShared_3489_; uint8_t v_isSharedCheck_3496_; 
v_one_3482_ = lean_unsigned_to_nat(1u);
v_n_3483_ = lean_nat_sub(v_x_3457_, v_one_3482_);
v___x_3484_ = lp_batteries_List_toChunksAux___redArg(v_n_3455_, v_tail_3461_, v_n_3483_);
lean_dec(v_n_3483_);
v_fst_3485_ = lean_ctor_get(v___x_3484_, 0);
v_snd_3486_ = lean_ctor_get(v___x_3484_, 1);
v_isSharedCheck_3496_ = !lean_is_exclusive(v___x_3484_);
if (v_isSharedCheck_3496_ == 0)
{
v___x_3488_ = v___x_3484_;
v_isShared_3489_ = v_isSharedCheck_3496_;
goto v_resetjp_3487_;
}
else
{
lean_inc(v_snd_3486_);
lean_inc(v_fst_3485_);
lean_dec(v___x_3484_);
v___x_3488_ = lean_box(0);
v_isShared_3489_ = v_isSharedCheck_3496_;
goto v_resetjp_3487_;
}
v_resetjp_3487_:
{
lean_object* v___x_3491_; 
if (v_isShared_3464_ == 0)
{
lean_ctor_set(v___x_3463_, 1, v_fst_3485_);
v___x_3491_ = v___x_3463_;
goto v_reusejp_3490_;
}
else
{
lean_object* v_reuseFailAlloc_3495_; 
v_reuseFailAlloc_3495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3495_, 0, v_head_3460_);
lean_ctor_set(v_reuseFailAlloc_3495_, 1, v_fst_3485_);
v___x_3491_ = v_reuseFailAlloc_3495_;
goto v_reusejp_3490_;
}
v_reusejp_3490_:
{
lean_object* v___x_3493_; 
if (v_isShared_3489_ == 0)
{
lean_ctor_set(v___x_3488_, 0, v___x_3491_);
v___x_3493_ = v___x_3488_;
goto v_reusejp_3492_;
}
else
{
lean_object* v_reuseFailAlloc_3494_; 
v_reuseFailAlloc_3494_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3494_, 0, v___x_3491_);
lean_ctor_set(v_reuseFailAlloc_3494_, 1, v_snd_3486_);
v___x_3493_ = v_reuseFailAlloc_3494_;
goto v_reusejp_3492_;
}
v_reusejp_3492_:
{
return v___x_3493_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunksAux___redArg___boxed(lean_object* v_n_3498_, lean_object* v_x_3499_, lean_object* v_x_3500_){
_start:
{
lean_object* v_res_3501_; 
v_res_3501_ = lp_batteries_List_toChunksAux___redArg(v_n_3498_, v_x_3499_, v_x_3500_);
lean_dec(v_x_3500_);
lean_dec(v_n_3498_);
return v_res_3501_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunksAux(lean_object* v_00_u03b1_3502_, lean_object* v_n_3503_, lean_object* v_x_3504_, lean_object* v_x_3505_){
_start:
{
lean_object* v___x_3506_; 
v___x_3506_ = lp_batteries_List_toChunksAux___redArg(v_n_3503_, v_x_3504_, v_x_3505_);
return v___x_3506_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunksAux___boxed(lean_object* v_00_u03b1_3507_, lean_object* v_n_3508_, lean_object* v_x_3509_, lean_object* v_x_3510_){
_start:
{
lean_object* v_res_3511_; 
v_res_3511_ = lp_batteries_List_toChunksAux(v_00_u03b1_3507_, v_n_3508_, v_x_3509_, v_x_3510_);
lean_dec(v_x_3510_);
lean_dec(v_n_3508_);
return v_res_3511_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunks_go___redArg(lean_object* v_n_3512_, lean_object* v_a_3513_, lean_object* v_a_3514_, lean_object* v_a_3515_){
_start:
{
if (lean_obj_tag(v_a_3513_) == 0)
{
lean_object* v___x_3516_; lean_object* v___x_3517_; lean_object* v___x_3518_; 
v___x_3516_ = lean_array_to_list(v_a_3514_);
v___x_3517_ = lean_array_push(v_a_3515_, v___x_3516_);
v___x_3518_ = lean_array_to_list(v___x_3517_);
return v___x_3518_;
}
else
{
lean_object* v_head_3519_; lean_object* v_tail_3520_; lean_object* v___x_3521_; uint8_t v___x_3522_; 
v_head_3519_ = lean_ctor_get(v_a_3513_, 0);
lean_inc(v_head_3519_);
v_tail_3520_ = lean_ctor_get(v_a_3513_, 1);
lean_inc(v_tail_3520_);
lean_dec_ref_known(v_a_3513_, 2);
v___x_3521_ = lean_array_get_size(v_a_3514_);
v___x_3522_ = lean_nat_dec_eq(v___x_3521_, v_n_3512_);
if (v___x_3522_ == 0)
{
lean_object* v___x_3523_; 
v___x_3523_ = lean_array_push(v_a_3514_, v_head_3519_);
v_a_3513_ = v_tail_3520_;
v_a_3514_ = v___x_3523_;
goto _start;
}
else
{
lean_object* v___x_3525_; lean_object* v___x_3526_; lean_object* v___x_3527_; lean_object* v___x_3528_; 
v___x_3525_ = lean_mk_empty_array_with_capacity(v_n_3512_);
v___x_3526_ = lean_array_push(v___x_3525_, v_head_3519_);
v___x_3527_ = lean_array_to_list(v_a_3514_);
v___x_3528_ = lean_array_push(v_a_3515_, v___x_3527_);
v_a_3513_ = v_tail_3520_;
v_a_3514_ = v___x_3526_;
v_a_3515_ = v___x_3528_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunks_go___redArg___boxed(lean_object* v_n_3530_, lean_object* v_a_3531_, lean_object* v_a_3532_, lean_object* v_a_3533_){
_start:
{
lean_object* v_res_3534_; 
v_res_3534_ = lp_batteries_List_toChunks_go___redArg(v_n_3530_, v_a_3531_, v_a_3532_, v_a_3533_);
lean_dec(v_n_3530_);
return v_res_3534_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunks_go(lean_object* v_00_u03b1_3535_, lean_object* v_n_3536_, lean_object* v_a_3537_, lean_object* v_a_3538_, lean_object* v_a_3539_){
_start:
{
lean_object* v___x_3540_; 
v___x_3540_ = lp_batteries_List_toChunks_go___redArg(v_n_3536_, v_a_3537_, v_a_3538_, v_a_3539_);
return v___x_3540_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunks_go___boxed(lean_object* v_00_u03b1_3541_, lean_object* v_n_3542_, lean_object* v_a_3543_, lean_object* v_a_3544_, lean_object* v_a_3545_){
_start:
{
lean_object* v_res_3546_; 
v_res_3546_ = lp_batteries_List_toChunks_go(v_00_u03b1_3541_, v_n_3542_, v_a_3543_, v_a_3544_, v_a_3545_);
lean_dec(v_n_3542_);
return v_res_3546_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunks___redArg(lean_object* v_x_3547_, lean_object* v_x_3548_){
_start:
{
if (lean_obj_tag(v_x_3548_) == 0)
{
lean_object* v___x_3549_; 
v___x_3549_ = lean_box(0);
return v___x_3549_;
}
else
{
lean_object* v_head_3550_; lean_object* v_tail_3551_; lean_object* v___x_3552_; uint8_t v___x_3553_; 
v_head_3550_ = lean_ctor_get(v_x_3548_, 0);
v_tail_3551_ = lean_ctor_get(v_x_3548_, 1);
v___x_3552_ = lean_unsigned_to_nat(0u);
v___x_3553_ = lean_nat_dec_eq(v_x_3547_, v___x_3552_);
if (v___x_3553_ == 0)
{
lean_object* v___x_3554_; lean_object* v___x_3555_; lean_object* v___x_3556_; lean_object* v___x_3557_; lean_object* v___x_3558_; 
lean_inc(v_tail_3551_);
lean_inc(v_head_3550_);
lean_dec_ref_known(v_x_3548_, 2);
v___x_3554_ = lean_unsigned_to_nat(1u);
v___x_3555_ = lean_mk_empty_array_with_capacity(v___x_3554_);
v___x_3556_ = lean_array_push(v___x_3555_, v_head_3550_);
v___x_3557_ = ((lean_object*)(lp_batteries_List_tailsTR___redArg___closed__0));
v___x_3558_ = lp_batteries_List_toChunks_go___redArg(v_x_3547_, v_tail_3551_, v___x_3556_, v___x_3557_);
return v___x_3558_;
}
else
{
lean_object* v___x_3559_; lean_object* v___x_3560_; 
v___x_3559_ = lean_box(0);
v___x_3560_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3560_, 0, v_x_3548_);
lean_ctor_set(v___x_3560_, 1, v___x_3559_);
return v___x_3560_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunks___redArg___boxed(lean_object* v_x_3561_, lean_object* v_x_3562_){
_start:
{
lean_object* v_res_3563_; 
v_res_3563_ = lp_batteries_List_toChunks___redArg(v_x_3561_, v_x_3562_);
lean_dec(v_x_3561_);
return v_res_3563_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunks(lean_object* v_00_u03b1_3564_, lean_object* v_x_3565_, lean_object* v_x_3566_){
_start:
{
lean_object* v___x_3567_; 
v___x_3567_ = lp_batteries_List_toChunks___redArg(v_x_3565_, v_x_3566_);
return v___x_3567_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_toChunks___boxed(lean_object* v_00_u03b1_3568_, lean_object* v_x_3569_, lean_object* v_x_3570_){
_start:
{
lean_object* v_res_3571_; 
v_res_3571_ = lp_batteries_List_toChunks(v_00_u03b1_3568_, v_x_3569_, v_x_3570_);
lean_dec(v_x_3569_);
return v_res_3571_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2083___redArg(lean_object* v_f_3572_, lean_object* v_x_3573_, lean_object* v_x_3574_, lean_object* v_x_3575_){
_start:
{
if (lean_obj_tag(v_x_3573_) == 1)
{
if (lean_obj_tag(v_x_3574_) == 1)
{
if (lean_obj_tag(v_x_3575_) == 1)
{
lean_object* v_head_3576_; lean_object* v_tail_3577_; lean_object* v_head_3578_; lean_object* v_tail_3579_; lean_object* v_head_3580_; lean_object* v_tail_3581_; lean_object* v___x_3583_; uint8_t v_isShared_3584_; uint8_t v_isSharedCheck_3590_; 
v_head_3576_ = lean_ctor_get(v_x_3573_, 0);
lean_inc(v_head_3576_);
v_tail_3577_ = lean_ctor_get(v_x_3573_, 1);
lean_inc(v_tail_3577_);
lean_dec_ref_known(v_x_3573_, 2);
v_head_3578_ = lean_ctor_get(v_x_3574_, 0);
lean_inc(v_head_3578_);
v_tail_3579_ = lean_ctor_get(v_x_3574_, 1);
lean_inc(v_tail_3579_);
lean_dec_ref_known(v_x_3574_, 2);
v_head_3580_ = lean_ctor_get(v_x_3575_, 0);
v_tail_3581_ = lean_ctor_get(v_x_3575_, 1);
v_isSharedCheck_3590_ = !lean_is_exclusive(v_x_3575_);
if (v_isSharedCheck_3590_ == 0)
{
v___x_3583_ = v_x_3575_;
v_isShared_3584_ = v_isSharedCheck_3590_;
goto v_resetjp_3582_;
}
else
{
lean_inc(v_tail_3581_);
lean_inc(v_head_3580_);
lean_dec(v_x_3575_);
v___x_3583_ = lean_box(0);
v_isShared_3584_ = v_isSharedCheck_3590_;
goto v_resetjp_3582_;
}
v_resetjp_3582_:
{
lean_object* v___x_3585_; lean_object* v___x_3586_; lean_object* v___x_3588_; 
lean_inc(v_f_3572_);
v___x_3585_ = lean_apply_3(v_f_3572_, v_head_3576_, v_head_3578_, v_head_3580_);
v___x_3586_ = lp_batteries_List_zipWith_u2083___redArg(v_f_3572_, v_tail_3577_, v_tail_3579_, v_tail_3581_);
if (v_isShared_3584_ == 0)
{
lean_ctor_set(v___x_3583_, 1, v___x_3586_);
lean_ctor_set(v___x_3583_, 0, v___x_3585_);
v___x_3588_ = v___x_3583_;
goto v_reusejp_3587_;
}
else
{
lean_object* v_reuseFailAlloc_3589_; 
v_reuseFailAlloc_3589_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3589_, 0, v___x_3585_);
lean_ctor_set(v_reuseFailAlloc_3589_, 1, v___x_3586_);
v___x_3588_ = v_reuseFailAlloc_3589_;
goto v_reusejp_3587_;
}
v_reusejp_3587_:
{
return v___x_3588_;
}
}
}
else
{
lean_object* v___x_3591_; 
lean_dec_ref_known(v_x_3574_, 2);
lean_dec_ref_known(v_x_3573_, 2);
lean_dec(v_x_3575_);
lean_dec(v_f_3572_);
v___x_3591_ = lean_box(0);
return v___x_3591_;
}
}
else
{
lean_object* v___x_3592_; 
lean_dec_ref_known(v_x_3573_, 2);
lean_dec(v_x_3575_);
lean_dec(v_x_3574_);
lean_dec(v_f_3572_);
v___x_3592_ = lean_box(0);
return v___x_3592_;
}
}
else
{
lean_object* v___x_3593_; 
lean_dec(v_x_3575_);
lean_dec(v_x_3574_);
lean_dec(v_x_3573_);
lean_dec(v_f_3572_);
v___x_3593_ = lean_box(0);
return v___x_3593_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2083(lean_object* v_00_u03b1_3594_, lean_object* v_00_u03b2_3595_, lean_object* v_00_u03b3_3596_, lean_object* v_00_u03b4_3597_, lean_object* v_f_3598_, lean_object* v_x_3599_, lean_object* v_x_3600_, lean_object* v_x_3601_){
_start:
{
lean_object* v___x_3602_; 
v___x_3602_ = lp_batteries_List_zipWith_u2083___redArg(v_f_3598_, v_x_3599_, v_x_3600_, v_x_3601_);
return v___x_3602_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2084___redArg(lean_object* v_f_3603_, lean_object* v_x_3604_, lean_object* v_x_3605_, lean_object* v_x_3606_, lean_object* v_x_3607_){
_start:
{
if (lean_obj_tag(v_x_3604_) == 1)
{
if (lean_obj_tag(v_x_3605_) == 1)
{
if (lean_obj_tag(v_x_3606_) == 1)
{
if (lean_obj_tag(v_x_3607_) == 1)
{
lean_object* v_head_3608_; lean_object* v_tail_3609_; lean_object* v_head_3610_; lean_object* v_tail_3611_; lean_object* v_head_3612_; lean_object* v_tail_3613_; lean_object* v_head_3614_; lean_object* v_tail_3615_; lean_object* v___x_3617_; uint8_t v_isShared_3618_; uint8_t v_isSharedCheck_3624_; 
v_head_3608_ = lean_ctor_get(v_x_3604_, 0);
lean_inc(v_head_3608_);
v_tail_3609_ = lean_ctor_get(v_x_3604_, 1);
lean_inc(v_tail_3609_);
lean_dec_ref_known(v_x_3604_, 2);
v_head_3610_ = lean_ctor_get(v_x_3605_, 0);
lean_inc(v_head_3610_);
v_tail_3611_ = lean_ctor_get(v_x_3605_, 1);
lean_inc(v_tail_3611_);
lean_dec_ref_known(v_x_3605_, 2);
v_head_3612_ = lean_ctor_get(v_x_3606_, 0);
lean_inc(v_head_3612_);
v_tail_3613_ = lean_ctor_get(v_x_3606_, 1);
lean_inc(v_tail_3613_);
lean_dec_ref_known(v_x_3606_, 2);
v_head_3614_ = lean_ctor_get(v_x_3607_, 0);
v_tail_3615_ = lean_ctor_get(v_x_3607_, 1);
v_isSharedCheck_3624_ = !lean_is_exclusive(v_x_3607_);
if (v_isSharedCheck_3624_ == 0)
{
v___x_3617_ = v_x_3607_;
v_isShared_3618_ = v_isSharedCheck_3624_;
goto v_resetjp_3616_;
}
else
{
lean_inc(v_tail_3615_);
lean_inc(v_head_3614_);
lean_dec(v_x_3607_);
v___x_3617_ = lean_box(0);
v_isShared_3618_ = v_isSharedCheck_3624_;
goto v_resetjp_3616_;
}
v_resetjp_3616_:
{
lean_object* v___x_3619_; lean_object* v___x_3620_; lean_object* v___x_3622_; 
lean_inc(v_f_3603_);
v___x_3619_ = lean_apply_4(v_f_3603_, v_head_3608_, v_head_3610_, v_head_3612_, v_head_3614_);
v___x_3620_ = lp_batteries_List_zipWith_u2084___redArg(v_f_3603_, v_tail_3609_, v_tail_3611_, v_tail_3613_, v_tail_3615_);
if (v_isShared_3618_ == 0)
{
lean_ctor_set(v___x_3617_, 1, v___x_3620_);
lean_ctor_set(v___x_3617_, 0, v___x_3619_);
v___x_3622_ = v___x_3617_;
goto v_reusejp_3621_;
}
else
{
lean_object* v_reuseFailAlloc_3623_; 
v_reuseFailAlloc_3623_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3623_, 0, v___x_3619_);
lean_ctor_set(v_reuseFailAlloc_3623_, 1, v___x_3620_);
v___x_3622_ = v_reuseFailAlloc_3623_;
goto v_reusejp_3621_;
}
v_reusejp_3621_:
{
return v___x_3622_;
}
}
}
else
{
lean_object* v___x_3625_; 
lean_dec_ref_known(v_x_3606_, 2);
lean_dec_ref_known(v_x_3605_, 2);
lean_dec_ref_known(v_x_3604_, 2);
lean_dec(v_x_3607_);
lean_dec(v_f_3603_);
v___x_3625_ = lean_box(0);
return v___x_3625_;
}
}
else
{
lean_object* v___x_3626_; 
lean_dec_ref_known(v_x_3605_, 2);
lean_dec_ref_known(v_x_3604_, 2);
lean_dec(v_x_3607_);
lean_dec(v_x_3606_);
lean_dec(v_f_3603_);
v___x_3626_ = lean_box(0);
return v___x_3626_;
}
}
else
{
lean_object* v___x_3627_; 
lean_dec_ref_known(v_x_3604_, 2);
lean_dec(v_x_3607_);
lean_dec(v_x_3606_);
lean_dec(v_x_3605_);
lean_dec(v_f_3603_);
v___x_3627_ = lean_box(0);
return v___x_3627_;
}
}
else
{
lean_object* v___x_3628_; 
lean_dec(v_x_3607_);
lean_dec(v_x_3606_);
lean_dec(v_x_3605_);
lean_dec(v_x_3604_);
lean_dec(v_f_3603_);
v___x_3628_ = lean_box(0);
return v___x_3628_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2084(lean_object* v_00_u03b1_3629_, lean_object* v_00_u03b2_3630_, lean_object* v_00_u03b3_3631_, lean_object* v_00_u03b4_3632_, lean_object* v_00_u03b5_3633_, lean_object* v_f_3634_, lean_object* v_x_3635_, lean_object* v_x_3636_, lean_object* v_x_3637_, lean_object* v_x_3638_){
_start:
{
lean_object* v___x_3639_; 
v___x_3639_ = lp_batteries_List_zipWith_u2084___redArg(v_f_3634_, v_x_3635_, v_x_3636_, v_x_3637_, v_x_3638_);
return v___x_3639_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2085___redArg(lean_object* v_f_3640_, lean_object* v_x_3641_, lean_object* v_x_3642_, lean_object* v_x_3643_, lean_object* v_x_3644_, lean_object* v_x_3645_){
_start:
{
if (lean_obj_tag(v_x_3641_) == 1)
{
if (lean_obj_tag(v_x_3642_) == 1)
{
if (lean_obj_tag(v_x_3643_) == 1)
{
if (lean_obj_tag(v_x_3644_) == 1)
{
if (lean_obj_tag(v_x_3645_) == 1)
{
lean_object* v_head_3646_; lean_object* v_tail_3647_; lean_object* v_head_3648_; lean_object* v_tail_3649_; lean_object* v_head_3650_; lean_object* v_tail_3651_; lean_object* v_head_3652_; lean_object* v_tail_3653_; lean_object* v_head_3654_; lean_object* v_tail_3655_; lean_object* v___x_3657_; uint8_t v_isShared_3658_; uint8_t v_isSharedCheck_3664_; 
v_head_3646_ = lean_ctor_get(v_x_3641_, 0);
lean_inc(v_head_3646_);
v_tail_3647_ = lean_ctor_get(v_x_3641_, 1);
lean_inc(v_tail_3647_);
lean_dec_ref_known(v_x_3641_, 2);
v_head_3648_ = lean_ctor_get(v_x_3642_, 0);
lean_inc(v_head_3648_);
v_tail_3649_ = lean_ctor_get(v_x_3642_, 1);
lean_inc(v_tail_3649_);
lean_dec_ref_known(v_x_3642_, 2);
v_head_3650_ = lean_ctor_get(v_x_3643_, 0);
lean_inc(v_head_3650_);
v_tail_3651_ = lean_ctor_get(v_x_3643_, 1);
lean_inc(v_tail_3651_);
lean_dec_ref_known(v_x_3643_, 2);
v_head_3652_ = lean_ctor_get(v_x_3644_, 0);
lean_inc(v_head_3652_);
v_tail_3653_ = lean_ctor_get(v_x_3644_, 1);
lean_inc(v_tail_3653_);
lean_dec_ref_known(v_x_3644_, 2);
v_head_3654_ = lean_ctor_get(v_x_3645_, 0);
v_tail_3655_ = lean_ctor_get(v_x_3645_, 1);
v_isSharedCheck_3664_ = !lean_is_exclusive(v_x_3645_);
if (v_isSharedCheck_3664_ == 0)
{
v___x_3657_ = v_x_3645_;
v_isShared_3658_ = v_isSharedCheck_3664_;
goto v_resetjp_3656_;
}
else
{
lean_inc(v_tail_3655_);
lean_inc(v_head_3654_);
lean_dec(v_x_3645_);
v___x_3657_ = lean_box(0);
v_isShared_3658_ = v_isSharedCheck_3664_;
goto v_resetjp_3656_;
}
v_resetjp_3656_:
{
lean_object* v___x_3659_; lean_object* v___x_3660_; lean_object* v___x_3662_; 
lean_inc(v_f_3640_);
v___x_3659_ = lean_apply_5(v_f_3640_, v_head_3646_, v_head_3648_, v_head_3650_, v_head_3652_, v_head_3654_);
v___x_3660_ = lp_batteries_List_zipWith_u2085___redArg(v_f_3640_, v_tail_3647_, v_tail_3649_, v_tail_3651_, v_tail_3653_, v_tail_3655_);
if (v_isShared_3658_ == 0)
{
lean_ctor_set(v___x_3657_, 1, v___x_3660_);
lean_ctor_set(v___x_3657_, 0, v___x_3659_);
v___x_3662_ = v___x_3657_;
goto v_reusejp_3661_;
}
else
{
lean_object* v_reuseFailAlloc_3663_; 
v_reuseFailAlloc_3663_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3663_, 0, v___x_3659_);
lean_ctor_set(v_reuseFailAlloc_3663_, 1, v___x_3660_);
v___x_3662_ = v_reuseFailAlloc_3663_;
goto v_reusejp_3661_;
}
v_reusejp_3661_:
{
return v___x_3662_;
}
}
}
else
{
lean_object* v___x_3665_; 
lean_dec_ref_known(v_x_3644_, 2);
lean_dec_ref_known(v_x_3643_, 2);
lean_dec_ref_known(v_x_3642_, 2);
lean_dec_ref_known(v_x_3641_, 2);
lean_dec(v_x_3645_);
lean_dec(v_f_3640_);
v___x_3665_ = lean_box(0);
return v___x_3665_;
}
}
else
{
lean_object* v___x_3666_; 
lean_dec_ref_known(v_x_3643_, 2);
lean_dec_ref_known(v_x_3642_, 2);
lean_dec_ref_known(v_x_3641_, 2);
lean_dec(v_x_3645_);
lean_dec(v_x_3644_);
lean_dec(v_f_3640_);
v___x_3666_ = lean_box(0);
return v___x_3666_;
}
}
else
{
lean_object* v___x_3667_; 
lean_dec_ref_known(v_x_3642_, 2);
lean_dec_ref_known(v_x_3641_, 2);
lean_dec(v_x_3645_);
lean_dec(v_x_3644_);
lean_dec(v_x_3643_);
lean_dec(v_f_3640_);
v___x_3667_ = lean_box(0);
return v___x_3667_;
}
}
else
{
lean_object* v___x_3668_; 
lean_dec_ref_known(v_x_3641_, 2);
lean_dec(v_x_3645_);
lean_dec(v_x_3644_);
lean_dec(v_x_3643_);
lean_dec(v_x_3642_);
lean_dec(v_f_3640_);
v___x_3668_ = lean_box(0);
return v___x_3668_;
}
}
else
{
lean_object* v___x_3669_; 
lean_dec(v_x_3645_);
lean_dec(v_x_3644_);
lean_dec(v_x_3643_);
lean_dec(v_x_3642_);
lean_dec(v_x_3641_);
lean_dec(v_f_3640_);
v___x_3669_ = lean_box(0);
return v___x_3669_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_zipWith_u2085(lean_object* v_00_u03b1_3670_, lean_object* v_00_u03b2_3671_, lean_object* v_00_u03b3_3672_, lean_object* v_00_u03b4_3673_, lean_object* v_00_u03b5_3674_, lean_object* v_00_u03b6_3675_, lean_object* v_f_3676_, lean_object* v_x_3677_, lean_object* v_x_3678_, lean_object* v_x_3679_, lean_object* v_x_3680_, lean_object* v_x_3681_){
_start:
{
lean_object* v___x_3682_; 
v___x_3682_ = lp_batteries_List_zipWith_u2085___redArg(v_f_3676_, v_x_3677_, v_x_3678_, v_x_3679_, v_x_3680_, v_x_3681_);
return v___x_3682_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapWithPrefixSuffixAux___redArg(lean_object* v_f_3683_, lean_object* v_x_3684_, lean_object* v_x_3685_){
_start:
{
if (lean_obj_tag(v_x_3685_) == 0)
{
lean_object* v___x_3686_; 
lean_dec(v_x_3684_);
lean_dec(v_f_3683_);
v___x_3686_ = lean_box(0);
return v___x_3686_;
}
else
{
lean_object* v_head_3687_; lean_object* v_tail_3688_; lean_object* v___x_3690_; uint8_t v_isShared_3691_; uint8_t v_isSharedCheck_3698_; 
v_head_3687_ = lean_ctor_get(v_x_3685_, 0);
v_tail_3688_ = lean_ctor_get(v_x_3685_, 1);
v_isSharedCheck_3698_ = !lean_is_exclusive(v_x_3685_);
if (v_isSharedCheck_3698_ == 0)
{
v___x_3690_ = v_x_3685_;
v_isShared_3691_ = v_isSharedCheck_3698_;
goto v_resetjp_3689_;
}
else
{
lean_inc(v_tail_3688_);
lean_inc(v_head_3687_);
lean_dec(v_x_3685_);
v___x_3690_ = lean_box(0);
v_isShared_3691_ = v_isSharedCheck_3698_;
goto v_resetjp_3689_;
}
v_resetjp_3689_:
{
lean_object* v___x_3692_; lean_object* v___x_3693_; lean_object* v___x_3694_; lean_object* v___x_3696_; 
lean_inc(v_f_3683_);
lean_inc(v_tail_3688_);
lean_inc(v_head_3687_);
lean_inc(v_x_3684_);
v___x_3692_ = lean_apply_3(v_f_3683_, v_x_3684_, v_head_3687_, v_tail_3688_);
v___x_3693_ = l_List_concat___redArg(v_x_3684_, v_head_3687_);
v___x_3694_ = lp_batteries_List_mapWithPrefixSuffixAux___redArg(v_f_3683_, v___x_3693_, v_tail_3688_);
if (v_isShared_3691_ == 0)
{
lean_ctor_set(v___x_3690_, 1, v___x_3694_);
lean_ctor_set(v___x_3690_, 0, v___x_3692_);
v___x_3696_ = v___x_3690_;
goto v_reusejp_3695_;
}
else
{
lean_object* v_reuseFailAlloc_3697_; 
v_reuseFailAlloc_3697_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3697_, 0, v___x_3692_);
lean_ctor_set(v_reuseFailAlloc_3697_, 1, v___x_3694_);
v___x_3696_ = v_reuseFailAlloc_3697_;
goto v_reusejp_3695_;
}
v_reusejp_3695_:
{
return v___x_3696_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapWithPrefixSuffixAux(lean_object* v_00_u03b1_3699_, lean_object* v_00_u03b2_3700_, lean_object* v_f_3701_, lean_object* v_x_3702_, lean_object* v_x_3703_){
_start:
{
lean_object* v___x_3704_; 
v___x_3704_ = lp_batteries_List_mapWithPrefixSuffixAux___redArg(v_f_3701_, v_x_3702_, v_x_3703_);
return v___x_3704_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapWithPrefixSuffix___redArg(lean_object* v_f_3705_, lean_object* v_l_3706_){
_start:
{
lean_object* v___x_3707_; lean_object* v___x_3708_; 
v___x_3707_ = lean_box(0);
v___x_3708_ = lp_batteries_List_mapWithPrefixSuffixAux___redArg(v_f_3705_, v___x_3707_, v_l_3706_);
return v___x_3708_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapWithPrefixSuffix(lean_object* v_00_u03b1_3709_, lean_object* v_00_u03b2_3710_, lean_object* v_f_3711_, lean_object* v_l_3712_){
_start:
{
lean_object* v___x_3713_; 
v___x_3713_ = lp_batteries_List_mapWithPrefixSuffix___redArg(v_f_3711_, v_l_3712_);
return v___x_3713_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapWithComplement___redArg___lam__0(lean_object* v_f_3714_, lean_object* v_pref_3715_, lean_object* v_a_3716_, lean_object* v_suff_3717_){
_start:
{
lean_object* v___x_3718_; lean_object* v___x_3719_; 
v___x_3718_ = l_List_appendTR___redArg(v_pref_3715_, v_suff_3717_);
v___x_3719_ = lean_apply_2(v_f_3714_, v_a_3716_, v___x_3718_);
return v___x_3719_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapWithComplement___redArg(lean_object* v_f_3720_, lean_object* v_l_3721_){
_start:
{
lean_object* v___f_3722_; lean_object* v___x_3723_; 
v___f_3722_ = lean_alloc_closure((void*)(lp_batteries_List_mapWithComplement___redArg___lam__0), 4, 1);
lean_closure_set(v___f_3722_, 0, v_f_3720_);
v___x_3723_ = lp_batteries_List_mapWithPrefixSuffix___redArg(v___f_3722_, v_l_3721_);
return v___x_3723_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapWithComplement(lean_object* v_00_u03b1_3724_, lean_object* v_00_u03b2_3725_, lean_object* v_f_3726_, lean_object* v_l_3727_){
_start:
{
lean_object* v___x_3728_; 
v___x_3728_ = lp_batteries_List_mapWithComplement___redArg(v_f_3726_, v_l_3727_);
return v___x_3728_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_traverse___redArg___lam__0(lean_object* v_head_3729_, lean_object* v_tail_3730_){
_start:
{
lean_object* v___x_3731_; 
v___x_3731_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3731_, 0, v_head_3729_);
lean_ctor_set(v___x_3731_, 1, v_tail_3730_);
return v___x_3731_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_traverse___redArg(lean_object* v_inst_3733_, lean_object* v_f_3734_, lean_object* v_x_3735_){
_start:
{
if (lean_obj_tag(v_x_3735_) == 0)
{
lean_object* v_toPure_3736_; lean_object* v___x_3737_; lean_object* v___x_3738_; 
lean_dec(v_f_3734_);
v_toPure_3736_ = lean_ctor_get(v_inst_3733_, 1);
lean_inc(v_toPure_3736_);
lean_dec_ref(v_inst_3733_);
v___x_3737_ = lean_box(0);
v___x_3738_ = lean_apply_2(v_toPure_3736_, lean_box(0), v___x_3737_);
return v___x_3738_;
}
else
{
lean_object* v_toFunctor_3739_; lean_object* v_toSeq_3740_; lean_object* v_head_3741_; lean_object* v_tail_3742_; lean_object* v_map_3743_; lean_object* v___f_3744_; lean_object* v___f_3745_; lean_object* v___x_3746_; lean_object* v___x_3747_; lean_object* v___x_3748_; 
v_toFunctor_3739_ = lean_ctor_get(v_inst_3733_, 0);
v_toSeq_3740_ = lean_ctor_get(v_inst_3733_, 2);
lean_inc(v_toSeq_3740_);
v_head_3741_ = lean_ctor_get(v_x_3735_, 0);
lean_inc(v_head_3741_);
v_tail_3742_ = lean_ctor_get(v_x_3735_, 1);
lean_inc(v_tail_3742_);
lean_dec_ref_known(v_x_3735_, 2);
v_map_3743_ = lean_ctor_get(v_toFunctor_3739_, 0);
lean_inc(v_map_3743_);
v___f_3744_ = ((lean_object*)(lp_batteries_List_traverse___redArg___closed__0));
lean_inc(v_f_3734_);
v___f_3745_ = lean_alloc_closure((void*)(lp_batteries_List_traverse___redArg___lam__1), 4, 3);
lean_closure_set(v___f_3745_, 0, v_inst_3733_);
lean_closure_set(v___f_3745_, 1, v_f_3734_);
lean_closure_set(v___f_3745_, 2, v_tail_3742_);
v___x_3746_ = lean_apply_1(v_f_3734_, v_head_3741_);
v___x_3747_ = lean_apply_4(v_map_3743_, lean_box(0), lean_box(0), v___f_3744_, v___x_3746_);
v___x_3748_ = lean_apply_4(v_toSeq_3740_, lean_box(0), lean_box(0), v___x_3747_, v___f_3745_);
return v___x_3748_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_traverse___redArg___lam__1(lean_object* v_inst_3749_, lean_object* v_f_3750_, lean_object* v_tail_3751_, lean_object* v_x_3752_){
_start:
{
lean_object* v___x_3753_; 
v___x_3753_ = lp_batteries_List_traverse___redArg(v_inst_3749_, v_f_3750_, v_tail_3751_);
return v___x_3753_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_traverse(lean_object* v_F_3754_, lean_object* v_00_u03b1_3755_, lean_object* v_00_u03b2_3756_, lean_object* v_inst_3757_, lean_object* v_f_3758_, lean_object* v_x_3759_){
_start:
{
lean_object* v___x_3760_; 
v___x_3760_ = lp_batteries_List_traverse___redArg(v_inst_3757_, v_f_3758_, v_x_3759_);
return v___x_3760_;
}
}
static lean_object* _init_lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__6(void){
_start:
{
lean_object* v___x_3797_; lean_object* v___x_3798_; 
v___x_3797_ = ((lean_object*)(lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__5));
v___x_3798_ = l_String_toRawSubstring_x27(v___x_3797_);
return v___x_3798_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1(lean_object* v_x_3813_, lean_object* v_a_3814_, lean_object* v_a_3815_){
_start:
{
lean_object* v___x_3816_; uint8_t v___x_3817_; 
v___x_3816_ = ((lean_object*)(lp_batteries_List_term___x3c_x2b_x7e___00__closed__2));
lean_inc(v_x_3813_);
v___x_3817_ = l_Lean_Syntax_isOfKind(v_x_3813_, v___x_3816_);
if (v___x_3817_ == 0)
{
lean_object* v___x_3818_; lean_object* v___x_3819_; 
lean_dec(v_x_3813_);
v___x_3818_ = lean_box(1);
v___x_3819_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3819_, 0, v___x_3818_);
lean_ctor_set(v___x_3819_, 1, v_a_3815_);
return v___x_3819_;
}
else
{
lean_object* v_quotContext_3820_; lean_object* v_currMacroScope_3821_; lean_object* v_ref_3822_; lean_object* v___x_3823_; lean_object* v___x_3824_; lean_object* v___x_3825_; lean_object* v___x_3826_; uint8_t v___x_3827_; lean_object* v___x_3828_; lean_object* v___x_3829_; lean_object* v___x_3830_; lean_object* v___x_3831_; lean_object* v___x_3832_; lean_object* v___x_3833_; lean_object* v___x_3834_; lean_object* v___x_3835_; lean_object* v___x_3836_; lean_object* v___x_3837_; lean_object* v___x_3838_; 
v_quotContext_3820_ = lean_ctor_get(v_a_3814_, 1);
v_currMacroScope_3821_ = lean_ctor_get(v_a_3814_, 2);
v_ref_3822_ = lean_ctor_get(v_a_3814_, 5);
v___x_3823_ = lean_unsigned_to_nat(0u);
v___x_3824_ = l_Lean_Syntax_getArg(v_x_3813_, v___x_3823_);
v___x_3825_ = lean_unsigned_to_nat(2u);
v___x_3826_ = l_Lean_Syntax_getArg(v_x_3813_, v___x_3825_);
lean_dec(v_x_3813_);
v___x_3827_ = 0;
v___x_3828_ = l_Lean_SourceInfo_fromRef(v_ref_3822_, v___x_3827_);
v___x_3829_ = ((lean_object*)(lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4));
v___x_3830_ = lean_obj_once(&lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__6, &lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__6_once, _init_lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__6);
v___x_3831_ = ((lean_object*)(lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__7));
lean_inc(v_currMacroScope_3821_);
lean_inc(v_quotContext_3820_);
v___x_3832_ = l_Lean_addMacroScope(v_quotContext_3820_, v___x_3831_, v_currMacroScope_3821_);
v___x_3833_ = ((lean_object*)(lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__10));
lean_inc_n(v___x_3828_, 2);
v___x_3834_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3834_, 0, v___x_3828_);
lean_ctor_set(v___x_3834_, 1, v___x_3830_);
lean_ctor_set(v___x_3834_, 2, v___x_3832_);
lean_ctor_set(v___x_3834_, 3, v___x_3833_);
v___x_3835_ = ((lean_object*)(lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__12));
v___x_3836_ = l_Lean_Syntax_node2(v___x_3828_, v___x_3835_, v___x_3824_, v___x_3826_);
v___x_3837_ = l_Lean_Syntax_node2(v___x_3828_, v___x_3829_, v___x_3834_, v___x_3836_);
v___x_3838_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3838_, 0, v___x_3837_);
lean_ctor_set(v___x_3838_, 1, v_a_3815_);
return v___x_3838_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___boxed(lean_object* v_x_3839_, lean_object* v_a_3840_, lean_object* v_a_3841_){
_start:
{
lean_object* v_res_3842_; 
v_res_3842_ = lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1(v_x_3839_, v_a_3840_, v_a_3841_);
lean_dec_ref(v_a_3840_);
return v_res_3842_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1(lean_object* v_x_3846_, lean_object* v_a_3847_, lean_object* v_a_3848_){
_start:
{
lean_object* v___x_3849_; uint8_t v___x_3850_; 
v___x_3849_ = ((lean_object*)(lp_batteries_List___aux__Batteries__Data__List__Basic______macroRules__List__term___x3c_x2b_x7e____1___closed__4));
lean_inc(v_x_3846_);
v___x_3850_ = l_Lean_Syntax_isOfKind(v_x_3846_, v___x_3849_);
if (v___x_3850_ == 0)
{
lean_object* v___x_3851_; lean_object* v___x_3852_; 
lean_dec(v_x_3846_);
v___x_3851_ = lean_box(0);
v___x_3852_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3852_, 0, v___x_3851_);
lean_ctor_set(v___x_3852_, 1, v_a_3848_);
return v___x_3852_;
}
else
{
lean_object* v___x_3853_; lean_object* v___x_3854_; lean_object* v___x_3855_; uint8_t v___x_3856_; 
v___x_3853_ = lean_unsigned_to_nat(0u);
v___x_3854_ = l_Lean_Syntax_getArg(v_x_3846_, v___x_3853_);
v___x_3855_ = ((lean_object*)(lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___closed__1));
lean_inc(v___x_3854_);
v___x_3856_ = l_Lean_Syntax_isOfKind(v___x_3854_, v___x_3855_);
if (v___x_3856_ == 0)
{
lean_object* v___x_3857_; lean_object* v___x_3858_; 
lean_dec(v___x_3854_);
lean_dec(v_x_3846_);
v___x_3857_ = lean_box(0);
v___x_3858_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3858_, 0, v___x_3857_);
lean_ctor_set(v___x_3858_, 1, v_a_3848_);
return v___x_3858_;
}
else
{
lean_object* v___x_3859_; lean_object* v___x_3860_; lean_object* v___x_3861_; uint8_t v___x_3862_; 
v___x_3859_ = lean_unsigned_to_nat(1u);
v___x_3860_ = l_Lean_Syntax_getArg(v_x_3846_, v___x_3859_);
lean_dec(v_x_3846_);
v___x_3861_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_3860_);
v___x_3862_ = l_Lean_Syntax_matchesNull(v___x_3860_, v___x_3861_);
if (v___x_3862_ == 0)
{
lean_object* v___x_3863_; lean_object* v___x_3864_; 
lean_dec(v___x_3860_);
lean_dec(v___x_3854_);
v___x_3863_ = lean_box(0);
v___x_3864_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3864_, 0, v___x_3863_);
lean_ctor_set(v___x_3864_, 1, v_a_3848_);
return v___x_3864_;
}
else
{
lean_object* v___x_3865_; lean_object* v___x_3866_; lean_object* v_ref_3867_; uint8_t v___x_3868_; lean_object* v___x_3869_; lean_object* v___x_3870_; lean_object* v___x_3871_; lean_object* v___x_3872_; lean_object* v___x_3873_; lean_object* v___x_3874_; 
v___x_3865_ = l_Lean_Syntax_getArg(v___x_3860_, v___x_3853_);
v___x_3866_ = l_Lean_Syntax_getArg(v___x_3860_, v___x_3859_);
lean_dec(v___x_3860_);
v_ref_3867_ = l_Lean_replaceRef(v___x_3854_, v_a_3847_);
lean_dec(v___x_3854_);
v___x_3868_ = 0;
v___x_3869_ = l_Lean_SourceInfo_fromRef(v_ref_3867_, v___x_3868_);
lean_dec(v_ref_3867_);
v___x_3870_ = ((lean_object*)(lp_batteries_List_term___x3c_x2b_x7e___00__closed__2));
v___x_3871_ = ((lean_object*)(lp_batteries_List_term___x3c_x2b_x7e___00__closed__5));
lean_inc(v___x_3869_);
v___x_3872_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3872_, 0, v___x_3869_);
lean_ctor_set(v___x_3872_, 1, v___x_3871_);
v___x_3873_ = l_Lean_Syntax_node3(v___x_3869_, v___x_3870_, v___x_3865_, v___x_3872_, v___x_3866_);
v___x_3874_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3874_, 0, v___x_3873_);
lean_ctor_set(v___x_3874_, 1, v_a_3848_);
return v___x_3874_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1___boxed(lean_object* v_x_3875_, lean_object* v_a_3876_, lean_object* v_a_3877_){
_start:
{
lean_object* v_res_3878_; 
v_res_3878_ = lp_batteries_List___aux__Batteries__Data__List__Basic______unexpand__List__Subperm__1(v_x_3875_, v_a_3876_, v_a_3877_);
lean_dec(v_a_3876_);
return v_res_3878_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_isSubperm___redArg___lam__2(lean_object* v_inst_3879_, lean_object* v_l_u2081_3880_, lean_object* v_l_u2082_3881_, lean_object* v_a_3882_){
_start:
{
lean_object* v___f_3883_; lean_object* v___x_3884_; lean_object* v___x_3885_; lean_object* v___x_3886_; uint8_t v___x_3887_; 
v___f_3883_ = lean_alloc_closure((void*)(lp_batteries_List_idxOfNth___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_3883_, 0, v_inst_3879_);
lean_closure_set(v___f_3883_, 1, v_a_3882_);
v___x_3884_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v___f_3883_);
v___x_3885_ = l_List_countP_go___redArg(v___f_3883_, v_l_u2081_3880_, v___x_3884_);
v___x_3886_ = l_List_countP_go___redArg(v___f_3883_, v_l_u2082_3881_, v___x_3884_);
v___x_3887_ = lean_nat_dec_le(v___x_3885_, v___x_3886_);
lean_dec(v___x_3886_);
lean_dec(v___x_3885_);
return v___x_3887_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_isSubperm___redArg___lam__2___boxed(lean_object* v_inst_3888_, lean_object* v_l_u2081_3889_, lean_object* v_l_u2082_3890_, lean_object* v_a_3891_){
_start:
{
uint8_t v_res_3892_; lean_object* v_r_3893_; 
v_res_3892_ = lp_batteries_List_isSubperm___redArg___lam__2(v_inst_3888_, v_l_u2081_3889_, v_l_u2082_3890_, v_a_3891_);
v_r_3893_ = lean_box(v_res_3892_);
return v_r_3893_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_isSubperm___redArg(lean_object* v_inst_3894_, lean_object* v_l_u2081_3895_, lean_object* v_l_u2082_3896_){
_start:
{
lean_object* v___f_3897_; uint8_t v___x_3898_; 
lean_inc(v_l_u2081_3895_);
v___f_3897_ = lean_alloc_closure((void*)(lp_batteries_List_isSubperm___redArg___lam__2___boxed), 4, 3);
lean_closure_set(v___f_3897_, 0, v_inst_3894_);
lean_closure_set(v___f_3897_, 1, v_l_u2081_3895_);
lean_closure_set(v___f_3897_, 2, v_l_u2082_3896_);
v___x_3898_ = l_List_decidableBAll___redArg(v___f_3897_, v_l_u2081_3895_);
return v___x_3898_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_isSubperm___redArg___boxed(lean_object* v_inst_3899_, lean_object* v_l_u2081_3900_, lean_object* v_l_u2082_3901_){
_start:
{
uint8_t v_res_3902_; lean_object* v_r_3903_; 
v_res_3902_ = lp_batteries_List_isSubperm___redArg(v_inst_3899_, v_l_u2081_3900_, v_l_u2082_3901_);
v_r_3903_ = lean_box(v_res_3902_);
return v_r_3903_;
}
}
LEAN_EXPORT uint8_t lp_batteries_List_isSubperm(lean_object* v_00_u03b1_3904_, lean_object* v_inst_3905_, lean_object* v_l_u2081_3906_, lean_object* v_l_u2082_3907_){
_start:
{
uint8_t v___x_3908_; 
v___x_3908_ = lp_batteries_List_isSubperm___redArg(v_inst_3905_, v_l_u2081_3906_, v_l_u2082_3907_);
return v___x_3908_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_isSubperm___boxed(lean_object* v_00_u03b1_3909_, lean_object* v_inst_3910_, lean_object* v_l_u2081_3911_, lean_object* v_l_u2082_3912_){
_start:
{
uint8_t v_res_3913_; lean_object* v_r_3914_; 
v_res_3913_ = lp_batteries_List_isSubperm(v_00_u03b1_3909_, v_inst_3910_, v_l_u2081_3911_, v_l_u2082_3912_);
v_r_3914_ = lean_box(v_res_3913_);
return v_r_3914_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_insertP_loop___redArg(lean_object* v_p_3915_, lean_object* v_a_3916_, lean_object* v_a_3917_, lean_object* v_a_3918_){
_start:
{
if (lean_obj_tag(v_a_3917_) == 0)
{
lean_object* v___x_3919_; lean_object* v___x_3920_; 
lean_dec_ref(v_p_3915_);
v___x_3919_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3919_, 0, v_a_3916_);
lean_ctor_set(v___x_3919_, 1, v_a_3918_);
v___x_3920_ = l_List_reverseAux___redArg(v___x_3919_, v_a_3917_);
return v___x_3920_;
}
else
{
lean_object* v_head_3921_; lean_object* v_tail_3922_; lean_object* v___x_3923_; uint8_t v___x_3924_; 
v_head_3921_ = lean_ctor_get(v_a_3917_, 0);
v_tail_3922_ = lean_ctor_get(v_a_3917_, 1);
lean_inc_ref(v_p_3915_);
lean_inc(v_head_3921_);
v___x_3923_ = lean_apply_1(v_p_3915_, v_head_3921_);
v___x_3924_ = lean_unbox(v___x_3923_);
if (v___x_3924_ == 0)
{
lean_object* v___x_3926_; uint8_t v_isShared_3927_; uint8_t v_isSharedCheck_3932_; 
lean_inc(v_tail_3922_);
lean_inc(v_head_3921_);
v_isSharedCheck_3932_ = !lean_is_exclusive(v_a_3917_);
if (v_isSharedCheck_3932_ == 0)
{
lean_object* v_unused_3933_; lean_object* v_unused_3934_; 
v_unused_3933_ = lean_ctor_get(v_a_3917_, 1);
lean_dec(v_unused_3933_);
v_unused_3934_ = lean_ctor_get(v_a_3917_, 0);
lean_dec(v_unused_3934_);
v___x_3926_ = v_a_3917_;
v_isShared_3927_ = v_isSharedCheck_3932_;
goto v_resetjp_3925_;
}
else
{
lean_dec(v_a_3917_);
v___x_3926_ = lean_box(0);
v_isShared_3927_ = v_isSharedCheck_3932_;
goto v_resetjp_3925_;
}
v_resetjp_3925_:
{
lean_object* v___x_3929_; 
if (v_isShared_3927_ == 0)
{
lean_ctor_set(v___x_3926_, 1, v_a_3918_);
v___x_3929_ = v___x_3926_;
goto v_reusejp_3928_;
}
else
{
lean_object* v_reuseFailAlloc_3931_; 
v_reuseFailAlloc_3931_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3931_, 0, v_head_3921_);
lean_ctor_set(v_reuseFailAlloc_3931_, 1, v_a_3918_);
v___x_3929_ = v_reuseFailAlloc_3931_;
goto v_reusejp_3928_;
}
v_reusejp_3928_:
{
v_a_3917_ = v_tail_3922_;
v_a_3918_ = v___x_3929_;
goto _start;
}
}
}
else
{
lean_object* v___x_3935_; lean_object* v___x_3936_; 
lean_dec_ref(v_p_3915_);
v___x_3935_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3935_, 0, v_a_3916_);
lean_ctor_set(v___x_3935_, 1, v_a_3918_);
v___x_3936_ = l_List_reverseAux___redArg(v___x_3935_, v_a_3917_);
return v___x_3936_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_insertP_loop(lean_object* v_00_u03b1_3937_, lean_object* v_p_3938_, lean_object* v_a_3939_, lean_object* v_a_3940_, lean_object* v_a_3941_){
_start:
{
lean_object* v___x_3942_; 
v___x_3942_ = lp_batteries_List_insertP_loop___redArg(v_p_3938_, v_a_3939_, v_a_3940_, v_a_3941_);
return v___x_3942_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_insertP___redArg(lean_object* v_p_3943_, lean_object* v_a_3944_, lean_object* v_l_3945_){
_start:
{
lean_object* v___x_3946_; lean_object* v___x_3947_; 
v___x_3946_ = lean_box(0);
v___x_3947_ = lp_batteries_List_insertP_loop___redArg(v_p_3943_, v_a_3944_, v_l_3945_, v___x_3946_);
return v___x_3947_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_insertP(lean_object* v_00_u03b1_3948_, lean_object* v_p_3949_, lean_object* v_a_3950_, lean_object* v_l_3951_){
_start:
{
lean_object* v___x_3952_; 
v___x_3952_ = lp_batteries_List_insertP___redArg(v_p_3949_, v_a_3950_, v_l_3951_);
return v___x_3952_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropPrefix_x3f___redArg(lean_object* v_inst_3953_, lean_object* v_x_3954_, lean_object* v_x_3955_){
_start:
{
if (lean_obj_tag(v_x_3955_) == 0)
{
lean_object* v___x_3956_; 
lean_dec_ref(v_inst_3953_);
v___x_3956_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3956_, 0, v_x_3954_);
return v___x_3956_;
}
else
{
if (lean_obj_tag(v_x_3954_) == 0)
{
lean_object* v___x_3957_; 
lean_dec_ref_known(v_x_3955_, 2);
lean_dec_ref(v_inst_3953_);
v___x_3957_ = lean_box(0);
return v___x_3957_;
}
else
{
lean_object* v_head_3958_; lean_object* v_tail_3959_; lean_object* v_head_3960_; lean_object* v_tail_3961_; lean_object* v___x_3962_; uint8_t v___x_3963_; 
v_head_3958_ = lean_ctor_get(v_x_3955_, 0);
lean_inc(v_head_3958_);
v_tail_3959_ = lean_ctor_get(v_x_3955_, 1);
lean_inc(v_tail_3959_);
lean_dec_ref_known(v_x_3955_, 2);
v_head_3960_ = lean_ctor_get(v_x_3954_, 0);
lean_inc(v_head_3960_);
v_tail_3961_ = lean_ctor_get(v_x_3954_, 1);
lean_inc(v_tail_3961_);
lean_dec_ref_known(v_x_3954_, 2);
lean_inc_ref(v_inst_3953_);
v___x_3962_ = lean_apply_2(v_inst_3953_, v_head_3960_, v_head_3958_);
v___x_3963_ = lean_unbox(v___x_3962_);
if (v___x_3963_ == 0)
{
lean_object* v___x_3964_; 
lean_dec(v_tail_3961_);
lean_dec(v_tail_3959_);
lean_dec_ref(v_inst_3953_);
v___x_3964_ = lean_box(0);
return v___x_3964_;
}
else
{
v_x_3954_ = v_tail_3961_;
v_x_3955_ = v_tail_3959_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropPrefix_x3f(lean_object* v_00_u03b1_3966_, lean_object* v_inst_3967_, lean_object* v_x_3968_, lean_object* v_x_3969_){
_start:
{
lean_object* v___x_3970_; 
v___x_3970_ = lp_batteries_List_dropPrefix_x3f___redArg(v_inst_3967_, v_x_3968_, v_x_3969_);
return v___x_3970_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSuffix_x3f___redArg(lean_object* v_inst_3971_, lean_object* v_l_3972_, lean_object* v_s_3973_){
_start:
{
lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; lean_object* v___x_3977_; lean_object* v_fst_3978_; lean_object* v_snd_3979_; uint8_t v___x_3980_; 
v___x_3974_ = l_List_lengthTR___redArg(v_l_3972_);
v___x_3975_ = l_List_lengthTR___redArg(v_s_3973_);
v___x_3976_ = lean_nat_sub(v___x_3974_, v___x_3975_);
lean_dec(v___x_3975_);
lean_dec(v___x_3974_);
v___x_3977_ = l_List_splitAt___redArg(v___x_3976_, v_l_3972_);
v_fst_3978_ = lean_ctor_get(v___x_3977_, 0);
lean_inc(v_fst_3978_);
v_snd_3979_ = lean_ctor_get(v___x_3977_, 1);
lean_inc(v_snd_3979_);
lean_dec_ref(v___x_3977_);
v___x_3980_ = l_List_beq___redArg(v_inst_3971_, v_snd_3979_, v_s_3973_);
if (v___x_3980_ == 0)
{
lean_object* v___x_3981_; 
lean_dec(v_fst_3978_);
v___x_3981_ = lean_box(0);
return v___x_3981_;
}
else
{
lean_object* v___x_3982_; 
v___x_3982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3982_, 0, v_fst_3978_);
return v___x_3982_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropSuffix_x3f(lean_object* v_00_u03b1_3983_, lean_object* v_inst_3984_, lean_object* v_l_3985_, lean_object* v_s_3986_){
_start:
{
lean_object* v___x_3987_; 
v___x_3987_ = lp_batteries_List_dropSuffix_x3f___redArg(v_inst_3984_, v_l_3985_, v_s_3986_);
return v___x_3987_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropInfix_x3f_go___redArg(lean_object* v_inst_3988_, lean_object* v_i_3989_, lean_object* v_a_3990_, lean_object* v_a_3991_){
_start:
{
if (lean_obj_tag(v_a_3990_) == 0)
{
uint8_t v___x_3992_; 
lean_dec_ref(v_inst_3988_);
v___x_3992_ = l_List_isEmpty___redArg(v_i_3989_);
lean_dec(v_i_3989_);
if (v___x_3992_ == 0)
{
lean_object* v___x_3993_; 
lean_dec(v_a_3991_);
v___x_3993_ = lean_box(0);
return v___x_3993_;
}
else
{
lean_object* v___x_3994_; lean_object* v___x_3995_; lean_object* v___x_3996_; 
v___x_3994_ = l_List_reverse___redArg(v_a_3991_);
v___x_3995_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3995_, 0, v___x_3994_);
lean_ctor_set(v___x_3995_, 1, v_a_3990_);
v___x_3996_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3996_, 0, v___x_3995_);
return v___x_3996_;
}
}
else
{
lean_object* v_head_3997_; lean_object* v_tail_3998_; lean_object* v___x_3999_; 
v_head_3997_ = lean_ctor_get(v_a_3990_, 0);
lean_inc(v_head_3997_);
v_tail_3998_ = lean_ctor_get(v_a_3990_, 1);
lean_inc(v_tail_3998_);
lean_inc(v_i_3989_);
lean_inc_ref(v_inst_3988_);
v___x_3999_ = lp_batteries_List_dropPrefix_x3f___redArg(v_inst_3988_, v_a_3990_, v_i_3989_);
if (lean_obj_tag(v___x_3999_) == 0)
{
lean_object* v___x_4000_; 
v___x_4000_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4000_, 0, v_head_3997_);
lean_ctor_set(v___x_4000_, 1, v_a_3991_);
v_a_3990_ = v_tail_3998_;
v_a_3991_ = v___x_4000_;
goto _start;
}
else
{
lean_object* v_val_4002_; lean_object* v___x_4004_; uint8_t v_isShared_4005_; uint8_t v_isSharedCheck_4011_; 
lean_dec(v_tail_3998_);
lean_dec(v_head_3997_);
lean_dec(v_i_3989_);
lean_dec_ref(v_inst_3988_);
v_val_4002_ = lean_ctor_get(v___x_3999_, 0);
v_isSharedCheck_4011_ = !lean_is_exclusive(v___x_3999_);
if (v_isSharedCheck_4011_ == 0)
{
v___x_4004_ = v___x_3999_;
v_isShared_4005_ = v_isSharedCheck_4011_;
goto v_resetjp_4003_;
}
else
{
lean_inc(v_val_4002_);
lean_dec(v___x_3999_);
v___x_4004_ = lean_box(0);
v_isShared_4005_ = v_isSharedCheck_4011_;
goto v_resetjp_4003_;
}
v_resetjp_4003_:
{
lean_object* v___x_4006_; lean_object* v___x_4007_; lean_object* v___x_4009_; 
v___x_4006_ = l_List_reverse___redArg(v_a_3991_);
v___x_4007_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4007_, 0, v___x_4006_);
lean_ctor_set(v___x_4007_, 1, v_val_4002_);
if (v_isShared_4005_ == 0)
{
lean_ctor_set(v___x_4004_, 0, v___x_4007_);
v___x_4009_ = v___x_4004_;
goto v_reusejp_4008_;
}
else
{
lean_object* v_reuseFailAlloc_4010_; 
v_reuseFailAlloc_4010_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4010_, 0, v___x_4007_);
v___x_4009_ = v_reuseFailAlloc_4010_;
goto v_reusejp_4008_;
}
v_reusejp_4008_:
{
return v___x_4009_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropInfix_x3f_go(lean_object* v_00_u03b1_4012_, lean_object* v_inst_4013_, lean_object* v_i_4014_, lean_object* v_a_4015_, lean_object* v_a_4016_){
_start:
{
lean_object* v___x_4017_; 
v___x_4017_ = lp_batteries_List_dropInfix_x3f_go___redArg(v_inst_4013_, v_i_4014_, v_a_4015_, v_a_4016_);
return v___x_4017_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropInfix_x3f___redArg(lean_object* v_inst_4018_, lean_object* v_l_4019_, lean_object* v_i_4020_){
_start:
{
lean_object* v___x_4021_; lean_object* v___x_4022_; 
v___x_4021_ = lean_box(0);
v___x_4022_ = lp_batteries_List_dropInfix_x3f_go___redArg(v_inst_4018_, v_i_4020_, v_l_4019_, v___x_4021_);
return v___x_4022_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_dropInfix_x3f(lean_object* v_00_u03b1_4023_, lean_object* v_inst_4024_, lean_object* v_l_4025_, lean_object* v_i_4026_){
_start:
{
lean_object* v___x_4027_; 
v___x_4027_ = lp_batteries_List_dropInfix_x3f___redArg(v_inst_4024_, v_l_4025_, v_i_4026_);
return v___x_4027_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_partialSums___redArg___lam__0(lean_object* v_inst_4028_, lean_object* v_x1_4029_, lean_object* v_x2_4030_){
_start:
{
lean_object* v___x_4031_; 
v___x_4031_ = lean_apply_2(v_inst_4028_, v_x1_4029_, v_x2_4030_);
return v___x_4031_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_partialSums___redArg(lean_object* v_inst_4032_, lean_object* v_inst_4033_, lean_object* v_l_4034_){
_start:
{
lean_object* v___f_4035_; lean_object* v___x_4036_; lean_object* v___x_4037_; lean_object* v___x_4038_; lean_object* v___x_4039_; 
v___f_4035_ = lean_alloc_closure((void*)(lp_batteries_List_partialSums___redArg___lam__0), 3, 1);
lean_closure_set(v___f_4035_, 0, v_inst_4032_);
v___x_4036_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__9));
v___x_4037_ = lean_box(0);
v___x_4038_ = l___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go(lean_box(0), lean_box(0), lean_box(0), v___x_4036_, v___f_4035_, v_l_4034_, v_inst_4033_, v___x_4037_);
v___x_4039_ = l_List_reverse___redArg(v___x_4038_);
return v___x_4039_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_partialSums(lean_object* v_00_u03b1_4040_, lean_object* v_inst_4041_, lean_object* v_inst_4042_, lean_object* v_l_4043_){
_start:
{
lean_object* v___x_4044_; 
v___x_4044_ = lp_batteries_List_partialSums___redArg(v_inst_4041_, v_inst_4042_, v_l_4043_);
return v___x_4044_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_partialProds___redArg(lean_object* v_inst_4045_, lean_object* v_inst_4046_, lean_object* v_l_4047_){
_start:
{
lean_object* v___f_4048_; lean_object* v___x_4049_; lean_object* v___x_4050_; lean_object* v___x_4051_; lean_object* v___x_4052_; 
v___f_4048_ = lean_alloc_closure((void*)(lp_batteries_List_partialSums___redArg___lam__0), 3, 1);
lean_closure_set(v___f_4048_, 0, v_inst_4045_);
v___x_4049_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__9));
v___x_4050_ = lean_box(0);
v___x_4051_ = l___private_Init_Data_List_Scan_Basic_0__List_scanAuxM_go(lean_box(0), lean_box(0), lean_box(0), v___x_4049_, v___f_4048_, v_l_4047_, v_inst_4046_, v___x_4050_);
v___x_4052_ = l_List_reverse___redArg(v___x_4051_);
return v___x_4052_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_partialProds(lean_object* v_00_u03b1_4053_, lean_object* v_inst_4054_, lean_object* v_inst_4055_, lean_object* v_l_4056_){
_start:
{
lean_object* v___x_4057_; 
v___x_4057_ = lp_batteries_List_partialProds___redArg(v_inst_4054_, v_inst_4055_, v_l_4056_);
return v___x_4057_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapAt___redArg(lean_object* v_xs_4058_, lean_object* v_i_4059_, lean_object* v_v_4060_){
_start:
{
lean_object* v___y_4062_; lean_object* v___x_4066_; 
lean_inc(v_i_4059_);
v___x_4066_ = l_List_get_x3fInternal___redArg(v_xs_4058_, v_i_4059_);
if (lean_obj_tag(v___x_4066_) == 0)
{
lean_inc(v_v_4060_);
v___y_4062_ = v_v_4060_;
goto v___jp_4061_;
}
else
{
lean_object* v_val_4067_; 
v_val_4067_ = lean_ctor_get(v___x_4066_, 0);
lean_inc(v_val_4067_);
lean_dec_ref_known(v___x_4066_, 1);
v___y_4062_ = v_val_4067_;
goto v___jp_4061_;
}
v___jp_4061_:
{
lean_object* v___x_4063_; lean_object* v___x_4064_; lean_object* v___x_4065_; 
v___x_4063_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_xs_4058_);
v___x_4064_ = l___private_Init_Data_List_Impl_0__List_setTR_go(lean_box(0), v_xs_4058_, v_v_4060_, v_xs_4058_, v_i_4059_, v___x_4063_);
lean_dec(v_xs_4058_);
v___x_4065_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4065_, 0, v___y_4062_);
lean_ctor_set(v___x_4065_, 1, v___x_4064_);
return v___x_4065_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapAt(lean_object* v_00_u03b1_4068_, lean_object* v_xs_4069_, lean_object* v_i_4070_, lean_object* v_v_4071_){
_start:
{
lean_object* v___x_4072_; 
v___x_4072_ = lp_batteries_List_swapAt___redArg(v_xs_4069_, v_i_4070_, v_v_4071_);
return v___x_4072_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapAtTR_go___redArg(lean_object* v_l_4073_, lean_object* v_v_4074_, lean_object* v_a_4075_, lean_object* v_a_4076_, lean_object* v_a_4077_){
_start:
{
if (lean_obj_tag(v_a_4075_) == 0)
{
lean_object* v___x_4078_; 
lean_dec_ref(v_a_4077_);
lean_dec(v_a_4076_);
v___x_4078_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4078_, 0, v_v_4074_);
lean_ctor_set(v___x_4078_, 1, v_l_4073_);
return v___x_4078_;
}
else
{
lean_object* v_head_4079_; lean_object* v_tail_4080_; lean_object* v___x_4082_; uint8_t v_isShared_4083_; uint8_t v_isSharedCheck_4102_; 
v_head_4079_ = lean_ctor_get(v_a_4075_, 0);
v_tail_4080_ = lean_ctor_get(v_a_4075_, 1);
v_isSharedCheck_4102_ = !lean_is_exclusive(v_a_4075_);
if (v_isSharedCheck_4102_ == 0)
{
v___x_4082_ = v_a_4075_;
v_isShared_4083_ = v_isSharedCheck_4102_;
goto v_resetjp_4081_;
}
else
{
lean_inc(v_tail_4080_);
lean_inc(v_head_4079_);
lean_dec(v_a_4075_);
v___x_4082_ = lean_box(0);
v_isShared_4083_ = v_isSharedCheck_4102_;
goto v_resetjp_4081_;
}
v_resetjp_4081_:
{
lean_object* v_zero_4084_; uint8_t v_isZero_4085_; 
v_zero_4084_ = lean_unsigned_to_nat(0u);
v_isZero_4085_ = lean_nat_dec_eq(v_a_4076_, v_zero_4084_);
if (v_isZero_4085_ == 1)
{
lean_object* v___x_4087_; 
lean_dec(v_a_4076_);
lean_dec(v_l_4073_);
if (v_isShared_4083_ == 0)
{
lean_ctor_set(v___x_4082_, 0, v_v_4074_);
v___x_4087_ = v___x_4082_;
goto v_reusejp_4086_;
}
else
{
lean_object* v_reuseFailAlloc_4097_; 
v_reuseFailAlloc_4097_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4097_, 0, v_v_4074_);
lean_ctor_set(v_reuseFailAlloc_4097_, 1, v_tail_4080_);
v___x_4087_ = v_reuseFailAlloc_4097_;
goto v_reusejp_4086_;
}
v_reusejp_4086_:
{
lean_object* v___x_4088_; lean_object* v___x_4089_; uint8_t v___x_4090_; 
v___x_4088_ = lean_array_get_size(v_a_4077_);
v___x_4089_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__9));
v___x_4090_ = lean_nat_dec_lt(v_zero_4084_, v___x_4088_);
if (v___x_4090_ == 0)
{
lean_object* v___x_4091_; 
lean_dec_ref(v_a_4077_);
v___x_4091_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4091_, 0, v_head_4079_);
lean_ctor_set(v___x_4091_, 1, v___x_4087_);
return v___x_4091_;
}
else
{
lean_object* v___f_4092_; size_t v___x_4093_; size_t v___x_4094_; lean_object* v___x_4095_; lean_object* v___x_4096_; 
v___f_4092_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__10));
v___x_4093_ = lean_usize_of_nat(v___x_4088_);
v___x_4094_ = ((size_t)0ULL);
v___x_4095_ = l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_4089_, v___f_4092_, v_a_4077_, v___x_4093_, v___x_4094_, v___x_4087_);
v___x_4096_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4096_, 0, v_head_4079_);
lean_ctor_set(v___x_4096_, 1, v___x_4095_);
return v___x_4096_;
}
}
}
else
{
lean_object* v_one_4098_; lean_object* v_n_4099_; lean_object* v___x_4100_; 
lean_del_object(v___x_4082_);
v_one_4098_ = lean_unsigned_to_nat(1u);
v_n_4099_ = lean_nat_sub(v_a_4076_, v_one_4098_);
lean_dec(v_a_4076_);
v___x_4100_ = lean_array_push(v_a_4077_, v_head_4079_);
v_a_4075_ = v_tail_4080_;
v_a_4076_ = v_n_4099_;
v_a_4077_ = v___x_4100_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapAtTR_go(lean_object* v_00_u03b1_4103_, lean_object* v_l_4104_, lean_object* v_v_4105_, lean_object* v_a_4106_, lean_object* v_a_4107_, lean_object* v_a_4108_){
_start:
{
lean_object* v___x_4109_; 
v___x_4109_ = lp_batteries_List_swapAtTR_go___redArg(v_l_4104_, v_v_4105_, v_a_4106_, v_a_4107_, v_a_4108_);
return v___x_4109_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapAtTR___redArg(lean_object* v_l_4110_, lean_object* v_i_4111_, lean_object* v_v_4112_){
_start:
{
lean_object* v___x_4113_; lean_object* v___x_4114_; 
v___x_4113_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_l_4110_);
v___x_4114_ = lp_batteries_List_swapAtTR_go___redArg(v_l_4110_, v_v_4112_, v_l_4110_, v_i_4111_, v___x_4113_);
return v___x_4114_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapAtTR(lean_object* v_00_u03b1_4115_, lean_object* v_l_4116_, lean_object* v_i_4117_, lean_object* v_v_4118_){
_start:
{
lean_object* v___x_4119_; lean_object* v___x_4120_; 
v___x_4119_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_l_4116_);
v___x_4120_ = lp_batteries_List_swapAtTR_go___redArg(v_l_4116_, v_v_4118_, v_l_4116_, v_i_4117_, v___x_4119_);
return v___x_4120_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swap___redArg(lean_object* v_x_4121_, lean_object* v_x_4122_, lean_object* v_x_4123_){
_start:
{
lean_object* v___y_4125_; lean_object* v___y_4126_; lean_object* v___y_4127_; lean_object* v___y_4128_; 
if (lean_obj_tag(v_x_4121_) == 0)
{
return v_x_4121_;
}
else
{
lean_object* v_head_4132_; lean_object* v_tail_4133_; lean_object* v_i_4135_; lean_object* v_zero_4138_; uint8_t v_isZero_4139_; 
v_head_4132_ = lean_ctor_get(v_x_4121_, 0);
v_tail_4133_ = lean_ctor_get(v_x_4121_, 1);
v_zero_4138_ = lean_unsigned_to_nat(0u);
v_isZero_4139_ = lean_nat_dec_eq(v_x_4122_, v_zero_4138_);
if (v_isZero_4139_ == 1)
{
uint8_t v_isZero_4140_; 
v_isZero_4140_ = lean_nat_dec_eq(v_x_4123_, v_zero_4138_);
if (v_isZero_4140_ == 1)
{
return v_x_4121_;
}
else
{
lean_object* v_one_4141_; lean_object* v_n_4142_; 
lean_inc(v_tail_4133_);
lean_inc(v_head_4132_);
lean_dec_ref_known(v_x_4121_, 2);
v_one_4141_ = lean_unsigned_to_nat(1u);
v_n_4142_ = lean_nat_sub(v_x_4123_, v_one_4141_);
v_i_4135_ = v_n_4142_;
goto v___jp_4134_;
}
}
else
{
lean_object* v___x_4144_; uint8_t v_isShared_4145_; uint8_t v_isSharedCheck_4154_; 
lean_inc(v_tail_4133_);
lean_inc(v_head_4132_);
v_isSharedCheck_4154_ = !lean_is_exclusive(v_x_4121_);
if (v_isSharedCheck_4154_ == 0)
{
lean_object* v_unused_4155_; lean_object* v_unused_4156_; 
v_unused_4155_ = lean_ctor_get(v_x_4121_, 1);
lean_dec(v_unused_4155_);
v_unused_4156_ = lean_ctor_get(v_x_4121_, 0);
lean_dec(v_unused_4156_);
v___x_4144_ = v_x_4121_;
v_isShared_4145_ = v_isSharedCheck_4154_;
goto v_resetjp_4143_;
}
else
{
lean_dec(v_x_4121_);
v___x_4144_ = lean_box(0);
v_isShared_4145_ = v_isSharedCheck_4154_;
goto v_resetjp_4143_;
}
v_resetjp_4143_:
{
lean_object* v_one_4146_; lean_object* v_n_4147_; uint8_t v_isZero_4148_; 
v_one_4146_ = lean_unsigned_to_nat(1u);
v_n_4147_ = lean_nat_sub(v_x_4122_, v_one_4146_);
v_isZero_4148_ = lean_nat_dec_eq(v_x_4123_, v_zero_4138_);
if (v_isZero_4148_ == 1)
{
lean_del_object(v___x_4144_);
v_i_4135_ = v_n_4147_;
goto v___jp_4134_;
}
else
{
lean_object* v_n_4149_; lean_object* v___x_4150_; lean_object* v___x_4152_; 
v_n_4149_ = lean_nat_sub(v_x_4123_, v_one_4146_);
v___x_4150_ = lp_batteries_List_swap___redArg(v_tail_4133_, v_n_4147_, v_n_4149_);
lean_dec(v_n_4149_);
lean_dec(v_n_4147_);
if (v_isShared_4145_ == 0)
{
lean_ctor_set(v___x_4144_, 1, v___x_4150_);
v___x_4152_ = v___x_4144_;
goto v_reusejp_4151_;
}
else
{
lean_object* v_reuseFailAlloc_4153_; 
v_reuseFailAlloc_4153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4153_, 0, v_head_4132_);
lean_ctor_set(v_reuseFailAlloc_4153_, 1, v___x_4150_);
v___x_4152_ = v_reuseFailAlloc_4153_;
goto v_reusejp_4151_;
}
v_reusejp_4151_:
{
return v___x_4152_;
}
}
}
}
v___jp_4134_:
{
lean_object* v___x_4136_; 
lean_inc(v_i_4135_);
v___x_4136_ = l_List_get_x3fInternal___redArg(v_tail_4133_, v_i_4135_);
if (lean_obj_tag(v___x_4136_) == 0)
{
lean_inc(v_head_4132_);
v___y_4125_ = v_tail_4133_;
v___y_4126_ = v_i_4135_;
v___y_4127_ = v_head_4132_;
v___y_4128_ = v_head_4132_;
goto v___jp_4124_;
}
else
{
lean_object* v_val_4137_; 
v_val_4137_ = lean_ctor_get(v___x_4136_, 0);
lean_inc(v_val_4137_);
lean_dec_ref_known(v___x_4136_, 1);
v___y_4125_ = v_tail_4133_;
v___y_4126_ = v_i_4135_;
v___y_4127_ = v_head_4132_;
v___y_4128_ = v_val_4137_;
goto v___jp_4124_;
}
}
}
v___jp_4124_:
{
lean_object* v___x_4129_; lean_object* v___x_4130_; lean_object* v___x_4131_; 
v___x_4129_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v___y_4125_);
v___x_4130_ = l___private_Init_Data_List_Impl_0__List_setTR_go(lean_box(0), v___y_4125_, v___y_4127_, v___y_4125_, v___y_4126_, v___x_4129_);
lean_dec(v___y_4125_);
v___x_4131_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4131_, 0, v___y_4128_);
lean_ctor_set(v___x_4131_, 1, v___x_4130_);
return v___x_4131_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swap___redArg___boxed(lean_object* v_x_4157_, lean_object* v_x_4158_, lean_object* v_x_4159_){
_start:
{
lean_object* v_res_4160_; 
v_res_4160_ = lp_batteries_List_swap___redArg(v_x_4157_, v_x_4158_, v_x_4159_);
lean_dec(v_x_4159_);
lean_dec(v_x_4158_);
return v_res_4160_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swap(lean_object* v_00_u03b1_4161_, lean_object* v_x_4162_, lean_object* v_x_4163_, lean_object* v_x_4164_){
_start:
{
lean_object* v___x_4165_; 
v___x_4165_ = lp_batteries_List_swap___redArg(v_x_4162_, v_x_4163_, v_x_4164_);
return v___x_4165_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swap___boxed(lean_object* v_00_u03b1_4166_, lean_object* v_x_4167_, lean_object* v_x_4168_, lean_object* v_x_4169_){
_start:
{
lean_object* v_res_4170_; 
v_res_4170_ = lp_batteries_List_swap(v_00_u03b1_4166_, v_x_4167_, v_x_4168_, v_x_4169_);
lean_dec(v_x_4169_);
lean_dec(v_x_4168_);
return v_res_4170_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapTR_go___redArg(lean_object* v_l_4171_, lean_object* v_a_4172_, lean_object* v_a_4173_, lean_object* v_a_4174_, lean_object* v_a_4175_){
_start:
{
if (lean_obj_tag(v_a_4172_) == 0)
{
lean_dec_ref(v_a_4175_);
lean_dec(v_a_4174_);
lean_dec(v_a_4173_);
lean_inc(v_l_4171_);
return v_l_4171_;
}
else
{
lean_object* v_head_4176_; lean_object* v_tail_4177_; lean_object* v___x_4179_; uint8_t v_isShared_4180_; uint8_t v_isSharedCheck_4210_; 
v_head_4176_ = lean_ctor_get(v_a_4172_, 0);
v_tail_4177_ = lean_ctor_get(v_a_4172_, 1);
v_isSharedCheck_4210_ = !lean_is_exclusive(v_a_4172_);
if (v_isSharedCheck_4210_ == 0)
{
v___x_4179_ = v_a_4172_;
v_isShared_4180_ = v_isSharedCheck_4210_;
goto v_resetjp_4178_;
}
else
{
lean_inc(v_tail_4177_);
lean_inc(v_head_4176_);
lean_dec(v_a_4172_);
v___x_4179_ = lean_box(0);
v_isShared_4180_ = v_isSharedCheck_4210_;
goto v_resetjp_4178_;
}
v_resetjp_4178_:
{
lean_object* v___f_4181_; lean_object* v_i_4183_; lean_object* v_acc_4184_; lean_object* v_zero_4199_; uint8_t v_isZero_4200_; 
v___f_4181_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__10));
v_zero_4199_ = lean_unsigned_to_nat(0u);
v_isZero_4200_ = lean_nat_dec_eq(v_a_4173_, v_zero_4199_);
if (v_isZero_4200_ == 1)
{
uint8_t v_isZero_4201_; 
lean_dec(v_a_4173_);
v_isZero_4201_ = lean_nat_dec_eq(v_a_4174_, v_zero_4199_);
if (v_isZero_4201_ == 1)
{
lean_del_object(v___x_4179_);
lean_dec(v_tail_4177_);
lean_dec(v_head_4176_);
lean_dec_ref(v_a_4175_);
lean_dec(v_a_4174_);
lean_inc(v_l_4171_);
return v_l_4171_;
}
else
{
lean_object* v_one_4202_; lean_object* v_n_4203_; 
v_one_4202_ = lean_unsigned_to_nat(1u);
v_n_4203_ = lean_nat_sub(v_a_4174_, v_one_4202_);
lean_dec(v_a_4174_);
v_i_4183_ = v_n_4203_;
v_acc_4184_ = v_a_4175_;
goto v___jp_4182_;
}
}
else
{
lean_object* v_one_4204_; lean_object* v_n_4205_; uint8_t v_isZero_4206_; 
v_one_4204_ = lean_unsigned_to_nat(1u);
v_n_4205_ = lean_nat_sub(v_a_4173_, v_one_4204_);
lean_dec(v_a_4173_);
v_isZero_4206_ = lean_nat_dec_eq(v_a_4174_, v_zero_4199_);
if (v_isZero_4206_ == 1)
{
lean_dec(v_a_4174_);
v_i_4183_ = v_n_4205_;
v_acc_4184_ = v_a_4175_;
goto v___jp_4182_;
}
else
{
lean_object* v_n_4207_; lean_object* v___x_4208_; 
lean_del_object(v___x_4179_);
v_n_4207_ = lean_nat_sub(v_a_4174_, v_one_4204_);
lean_dec(v_a_4174_);
v___x_4208_ = lean_array_push(v_a_4175_, v_head_4176_);
v_a_4172_ = v_tail_4177_;
v_a_4173_ = v_n_4205_;
v_a_4174_ = v_n_4207_;
v_a_4175_ = v___x_4208_;
goto _start;
}
}
v___jp_4182_:
{
lean_object* v___x_4185_; lean_object* v___x_4186_; lean_object* v___x_4187_; lean_object* v_fst_4188_; lean_object* v_snd_4189_; lean_object* v___x_4191_; 
v___x_4185_ = lean_unsigned_to_nat(0u);
v___x_4186_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_tail_4177_);
v___x_4187_ = lp_batteries_List_swapAtTR_go___redArg(v_tail_4177_, v_head_4176_, v_tail_4177_, v_i_4183_, v___x_4186_);
v_fst_4188_ = lean_ctor_get(v___x_4187_, 0);
lean_inc(v_fst_4188_);
v_snd_4189_ = lean_ctor_get(v___x_4187_, 1);
lean_inc(v_snd_4189_);
lean_dec_ref(v___x_4187_);
if (v_isShared_4180_ == 0)
{
lean_ctor_set(v___x_4179_, 1, v_snd_4189_);
lean_ctor_set(v___x_4179_, 0, v_fst_4188_);
v___x_4191_ = v___x_4179_;
goto v_reusejp_4190_;
}
else
{
lean_object* v_reuseFailAlloc_4198_; 
v_reuseFailAlloc_4198_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4198_, 0, v_fst_4188_);
lean_ctor_set(v_reuseFailAlloc_4198_, 1, v_snd_4189_);
v___x_4191_ = v_reuseFailAlloc_4198_;
goto v_reusejp_4190_;
}
v_reusejp_4190_:
{
lean_object* v___x_4192_; lean_object* v___x_4193_; uint8_t v___x_4194_; 
v___x_4192_ = lean_array_get_size(v_acc_4184_);
v___x_4193_ = ((lean_object*)(lp_batteries_List_replaceFTR_go___redArg___closed__9));
v___x_4194_ = lean_nat_dec_lt(v___x_4185_, v___x_4192_);
if (v___x_4194_ == 0)
{
lean_dec_ref(v_acc_4184_);
return v___x_4191_;
}
else
{
size_t v___x_4195_; size_t v___x_4196_; lean_object* v___x_4197_; 
v___x_4195_ = lean_usize_of_nat(v___x_4192_);
v___x_4196_ = ((size_t)0ULL);
v___x_4197_ = l___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_4193_, v___f_4181_, v_acc_4184_, v___x_4195_, v___x_4196_, v___x_4191_);
return v___x_4197_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapTR_go___redArg___boxed(lean_object* v_l_4211_, lean_object* v_a_4212_, lean_object* v_a_4213_, lean_object* v_a_4214_, lean_object* v_a_4215_){
_start:
{
lean_object* v_res_4216_; 
v_res_4216_ = lp_batteries_List_swapTR_go___redArg(v_l_4211_, v_a_4212_, v_a_4213_, v_a_4214_, v_a_4215_);
lean_dec(v_l_4211_);
return v_res_4216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapTR_go(lean_object* v_00_u03b1_4217_, lean_object* v_l_4218_, lean_object* v_a_4219_, lean_object* v_a_4220_, lean_object* v_a_4221_, lean_object* v_a_4222_){
_start:
{
lean_object* v___x_4223_; 
v___x_4223_ = lp_batteries_List_swapTR_go___redArg(v_l_4218_, v_a_4219_, v_a_4220_, v_a_4221_, v_a_4222_);
return v___x_4223_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapTR_go___boxed(lean_object* v_00_u03b1_4224_, lean_object* v_l_4225_, lean_object* v_a_4226_, lean_object* v_a_4227_, lean_object* v_a_4228_, lean_object* v_a_4229_){
_start:
{
lean_object* v_res_4230_; 
v_res_4230_ = lp_batteries_List_swapTR_go(v_00_u03b1_4224_, v_l_4225_, v_a_4226_, v_a_4227_, v_a_4228_, v_a_4229_);
lean_dec(v_l_4225_);
return v_res_4230_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapTR___redArg(lean_object* v_l_4231_, lean_object* v_i_4232_, lean_object* v_j_4233_){
_start:
{
lean_object* v___x_4234_; lean_object* v___x_4235_; 
v___x_4234_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_l_4231_);
v___x_4235_ = lp_batteries_List_swapTR_go___redArg(v_l_4231_, v_l_4231_, v_i_4232_, v_j_4233_, v___x_4234_);
lean_dec(v_l_4231_);
return v___x_4235_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_swapTR(lean_object* v_00_u03b1_4236_, lean_object* v_l_4237_, lean_object* v_i_4238_, lean_object* v_j_4239_){
_start:
{
lean_object* v___x_4240_; lean_object* v___x_4241_; 
v___x_4240_ = ((lean_object*)(lp_batteries_List_bagInter___redArg___closed__0));
lean_inc(v_l_4237_);
v___x_4241_ = lp_batteries_List_swapTR_go___redArg(v_l_4237_, v_l_4237_, v_i_4238_, v_j_4239_, v___x_4240_);
lean_dec(v_l_4237_);
return v___x_4241_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_swapTR_go_match__1_splitter___redArg(lean_object* v_x_4242_, lean_object* v_x_4243_, lean_object* v_x_4244_, lean_object* v_x_4245_, lean_object* v_h__1_4246_, lean_object* v_h__2_4247_, lean_object* v_h__3_4248_, lean_object* v_h__4_4249_, lean_object* v_h__5_4250_){
_start:
{
if (lean_obj_tag(v_x_4242_) == 0)
{
lean_object* v___x_4251_; 
lean_dec(v_h__5_4250_);
lean_dec(v_h__4_4249_);
lean_dec(v_h__3_4248_);
lean_dec(v_h__2_4247_);
v___x_4251_ = lean_apply_3(v_h__1_4246_, v_x_4243_, v_x_4244_, v_x_4245_);
return v___x_4251_;
}
else
{
lean_object* v_head_4252_; lean_object* v_tail_4253_; lean_object* v_zero_4254_; uint8_t v_isZero_4255_; 
lean_dec(v_h__1_4246_);
v_head_4252_ = lean_ctor_get(v_x_4242_, 0);
lean_inc(v_head_4252_);
v_tail_4253_ = lean_ctor_get(v_x_4242_, 1);
lean_inc(v_tail_4253_);
lean_dec_ref_known(v_x_4242_, 2);
v_zero_4254_ = lean_unsigned_to_nat(0u);
v_isZero_4255_ = lean_nat_dec_eq(v_x_4243_, v_zero_4254_);
if (v_isZero_4255_ == 1)
{
uint8_t v_isZero_4256_; 
lean_dec(v_h__5_4250_);
lean_dec(v_h__4_4249_);
lean_dec(v_x_4243_);
v_isZero_4256_ = lean_nat_dec_eq(v_x_4244_, v_zero_4254_);
if (v_isZero_4256_ == 1)
{
lean_object* v___x_4257_; 
lean_dec(v_h__3_4248_);
lean_dec(v_x_4244_);
v___x_4257_ = lean_apply_3(v_h__2_4247_, v_head_4252_, v_tail_4253_, v_x_4245_);
return v___x_4257_;
}
else
{
lean_object* v_one_4258_; lean_object* v_n_4259_; lean_object* v___x_4260_; 
lean_dec(v_h__2_4247_);
v_one_4258_ = lean_unsigned_to_nat(1u);
v_n_4259_ = lean_nat_sub(v_x_4244_, v_one_4258_);
lean_dec(v_x_4244_);
v___x_4260_ = lean_apply_4(v_h__3_4248_, v_head_4252_, v_tail_4253_, v_n_4259_, v_x_4245_);
return v___x_4260_;
}
}
else
{
lean_object* v_one_4261_; lean_object* v_n_4262_; uint8_t v_isZero_4263_; 
lean_dec(v_h__3_4248_);
lean_dec(v_h__2_4247_);
v_one_4261_ = lean_unsigned_to_nat(1u);
v_n_4262_ = lean_nat_sub(v_x_4243_, v_one_4261_);
lean_dec(v_x_4243_);
v_isZero_4263_ = lean_nat_dec_eq(v_x_4244_, v_zero_4254_);
if (v_isZero_4263_ == 1)
{
lean_object* v___x_4264_; 
lean_dec(v_h__5_4250_);
lean_dec(v_x_4244_);
v___x_4264_ = lean_apply_4(v_h__4_4249_, v_head_4252_, v_tail_4253_, v_n_4262_, v_x_4245_);
return v___x_4264_;
}
else
{
lean_object* v_n_4265_; lean_object* v___x_4266_; 
lean_dec(v_h__4_4249_);
v_n_4265_ = lean_nat_sub(v_x_4244_, v_one_4261_);
lean_dec(v_x_4244_);
v___x_4266_ = lean_apply_5(v_h__5_4250_, v_head_4252_, v_tail_4253_, v_n_4262_, v_n_4265_, v_x_4245_);
return v___x_4266_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_swapTR_go_match__1_splitter(lean_object* v_00_u03b1_4267_, lean_object* v_motive_4268_, lean_object* v_x_4269_, lean_object* v_x_4270_, lean_object* v_x_4271_, lean_object* v_x_4272_, lean_object* v_h__1_4273_, lean_object* v_h__2_4274_, lean_object* v_h__3_4275_, lean_object* v_h__4_4276_, lean_object* v_h__5_4277_){
_start:
{
if (lean_obj_tag(v_x_4269_) == 0)
{
lean_object* v___x_4278_; 
lean_dec(v_h__5_4277_);
lean_dec(v_h__4_4276_);
lean_dec(v_h__3_4275_);
lean_dec(v_h__2_4274_);
v___x_4278_ = lean_apply_3(v_h__1_4273_, v_x_4270_, v_x_4271_, v_x_4272_);
return v___x_4278_;
}
else
{
lean_object* v_head_4279_; lean_object* v_tail_4280_; lean_object* v_zero_4281_; uint8_t v_isZero_4282_; 
lean_dec(v_h__1_4273_);
v_head_4279_ = lean_ctor_get(v_x_4269_, 0);
lean_inc(v_head_4279_);
v_tail_4280_ = lean_ctor_get(v_x_4269_, 1);
lean_inc(v_tail_4280_);
lean_dec_ref_known(v_x_4269_, 2);
v_zero_4281_ = lean_unsigned_to_nat(0u);
v_isZero_4282_ = lean_nat_dec_eq(v_x_4270_, v_zero_4281_);
if (v_isZero_4282_ == 1)
{
uint8_t v_isZero_4283_; 
lean_dec(v_h__5_4277_);
lean_dec(v_h__4_4276_);
lean_dec(v_x_4270_);
v_isZero_4283_ = lean_nat_dec_eq(v_x_4271_, v_zero_4281_);
if (v_isZero_4283_ == 1)
{
lean_object* v___x_4284_; 
lean_dec(v_h__3_4275_);
lean_dec(v_x_4271_);
v___x_4284_ = lean_apply_3(v_h__2_4274_, v_head_4279_, v_tail_4280_, v_x_4272_);
return v___x_4284_;
}
else
{
lean_object* v_one_4285_; lean_object* v_n_4286_; lean_object* v___x_4287_; 
lean_dec(v_h__2_4274_);
v_one_4285_ = lean_unsigned_to_nat(1u);
v_n_4286_ = lean_nat_sub(v_x_4271_, v_one_4285_);
lean_dec(v_x_4271_);
v___x_4287_ = lean_apply_4(v_h__3_4275_, v_head_4279_, v_tail_4280_, v_n_4286_, v_x_4272_);
return v___x_4287_;
}
}
else
{
lean_object* v_one_4288_; lean_object* v_n_4289_; uint8_t v_isZero_4290_; 
lean_dec(v_h__3_4275_);
lean_dec(v_h__2_4274_);
v_one_4288_ = lean_unsigned_to_nat(1u);
v_n_4289_ = lean_nat_sub(v_x_4270_, v_one_4288_);
lean_dec(v_x_4270_);
v_isZero_4290_ = lean_nat_dec_eq(v_x_4271_, v_zero_4281_);
if (v_isZero_4290_ == 1)
{
lean_object* v___x_4291_; 
lean_dec(v_h__5_4277_);
lean_dec(v_x_4271_);
v___x_4291_ = lean_apply_4(v_h__4_4276_, v_head_4279_, v_tail_4280_, v_n_4289_, v_x_4272_);
return v___x_4291_;
}
else
{
lean_object* v_n_4292_; lean_object* v___x_4293_; 
lean_dec(v_h__4_4276_);
v_n_4292_ = lean_nat_sub(v_x_4271_, v_one_4288_);
lean_dec(v_x_4271_);
v___x_4293_ = lean_apply_5(v_h__5_4277_, v_head_4279_, v_tail_4280_, v_n_4289_, v_n_4292_, v_x_4272_);
return v___x_4293_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_swap_match__1_splitter___redArg(lean_object* v_x_4294_, lean_object* v_x_4295_, lean_object* v_x_4296_, lean_object* v_h__1_4297_, lean_object* v_h__2_4298_, lean_object* v_h__3_4299_, lean_object* v_h__4_4300_, lean_object* v_h__5_4301_){
_start:
{
if (lean_obj_tag(v_x_4294_) == 0)
{
lean_object* v___x_4302_; 
lean_dec(v_h__5_4301_);
lean_dec(v_h__4_4300_);
lean_dec(v_h__3_4299_);
lean_dec(v_h__2_4298_);
v___x_4302_ = lean_apply_2(v_h__1_4297_, v_x_4295_, v_x_4296_);
return v___x_4302_;
}
else
{
lean_object* v_head_4303_; lean_object* v_tail_4304_; lean_object* v_zero_4305_; uint8_t v_isZero_4306_; 
lean_dec(v_h__1_4297_);
v_head_4303_ = lean_ctor_get(v_x_4294_, 0);
lean_inc(v_head_4303_);
v_tail_4304_ = lean_ctor_get(v_x_4294_, 1);
lean_inc(v_tail_4304_);
lean_dec_ref_known(v_x_4294_, 2);
v_zero_4305_ = lean_unsigned_to_nat(0u);
v_isZero_4306_ = lean_nat_dec_eq(v_x_4295_, v_zero_4305_);
if (v_isZero_4306_ == 1)
{
uint8_t v_isZero_4307_; 
lean_dec(v_h__5_4301_);
lean_dec(v_h__4_4300_);
lean_dec(v_x_4295_);
v_isZero_4307_ = lean_nat_dec_eq(v_x_4296_, v_zero_4305_);
if (v_isZero_4307_ == 1)
{
lean_object* v___x_4308_; 
lean_dec(v_h__3_4299_);
lean_dec(v_x_4296_);
v___x_4308_ = lean_apply_2(v_h__2_4298_, v_head_4303_, v_tail_4304_);
return v___x_4308_;
}
else
{
lean_object* v_one_4309_; lean_object* v_n_4310_; lean_object* v___x_4311_; 
lean_dec(v_h__2_4298_);
v_one_4309_ = lean_unsigned_to_nat(1u);
v_n_4310_ = lean_nat_sub(v_x_4296_, v_one_4309_);
lean_dec(v_x_4296_);
v___x_4311_ = lean_apply_3(v_h__3_4299_, v_head_4303_, v_tail_4304_, v_n_4310_);
return v___x_4311_;
}
}
else
{
lean_object* v_one_4312_; lean_object* v_n_4313_; uint8_t v_isZero_4314_; 
lean_dec(v_h__3_4299_);
lean_dec(v_h__2_4298_);
v_one_4312_ = lean_unsigned_to_nat(1u);
v_n_4313_ = lean_nat_sub(v_x_4295_, v_one_4312_);
lean_dec(v_x_4295_);
v_isZero_4314_ = lean_nat_dec_eq(v_x_4296_, v_zero_4305_);
if (v_isZero_4314_ == 1)
{
lean_object* v___x_4315_; 
lean_dec(v_h__5_4301_);
lean_dec(v_x_4296_);
v___x_4315_ = lean_apply_3(v_h__4_4300_, v_head_4303_, v_tail_4304_, v_n_4313_);
return v___x_4315_;
}
else
{
lean_object* v_n_4316_; lean_object* v___x_4317_; 
lean_dec(v_h__4_4300_);
v_n_4316_ = lean_nat_sub(v_x_4296_, v_one_4312_);
lean_dec(v_x_4296_);
v___x_4317_ = lean_apply_4(v_h__5_4301_, v_head_4303_, v_tail_4304_, v_n_4313_, v_n_4316_);
return v___x_4317_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_List_Basic_0__List_swap_match__1_splitter(lean_object* v_00_u03b1_4318_, lean_object* v_motive_4319_, lean_object* v_x_4320_, lean_object* v_x_4321_, lean_object* v_x_4322_, lean_object* v_h__1_4323_, lean_object* v_h__2_4324_, lean_object* v_h__3_4325_, lean_object* v_h__4_4326_, lean_object* v_h__5_4327_){
_start:
{
if (lean_obj_tag(v_x_4320_) == 0)
{
lean_object* v___x_4328_; 
lean_dec(v_h__5_4327_);
lean_dec(v_h__4_4326_);
lean_dec(v_h__3_4325_);
lean_dec(v_h__2_4324_);
v___x_4328_ = lean_apply_2(v_h__1_4323_, v_x_4321_, v_x_4322_);
return v___x_4328_;
}
else
{
lean_object* v_head_4329_; lean_object* v_tail_4330_; lean_object* v_zero_4331_; uint8_t v_isZero_4332_; 
lean_dec(v_h__1_4323_);
v_head_4329_ = lean_ctor_get(v_x_4320_, 0);
lean_inc(v_head_4329_);
v_tail_4330_ = lean_ctor_get(v_x_4320_, 1);
lean_inc(v_tail_4330_);
lean_dec_ref_known(v_x_4320_, 2);
v_zero_4331_ = lean_unsigned_to_nat(0u);
v_isZero_4332_ = lean_nat_dec_eq(v_x_4321_, v_zero_4331_);
if (v_isZero_4332_ == 1)
{
uint8_t v_isZero_4333_; 
lean_dec(v_h__5_4327_);
lean_dec(v_h__4_4326_);
lean_dec(v_x_4321_);
v_isZero_4333_ = lean_nat_dec_eq(v_x_4322_, v_zero_4331_);
if (v_isZero_4333_ == 1)
{
lean_object* v___x_4334_; 
lean_dec(v_h__3_4325_);
lean_dec(v_x_4322_);
v___x_4334_ = lean_apply_2(v_h__2_4324_, v_head_4329_, v_tail_4330_);
return v___x_4334_;
}
else
{
lean_object* v_one_4335_; lean_object* v_n_4336_; lean_object* v___x_4337_; 
lean_dec(v_h__2_4324_);
v_one_4335_ = lean_unsigned_to_nat(1u);
v_n_4336_ = lean_nat_sub(v_x_4322_, v_one_4335_);
lean_dec(v_x_4322_);
v___x_4337_ = lean_apply_3(v_h__3_4325_, v_head_4329_, v_tail_4330_, v_n_4336_);
return v___x_4337_;
}
}
else
{
lean_object* v_one_4338_; lean_object* v_n_4339_; uint8_t v_isZero_4340_; 
lean_dec(v_h__3_4325_);
lean_dec(v_h__2_4324_);
v_one_4338_ = lean_unsigned_to_nat(1u);
v_n_4339_ = lean_nat_sub(v_x_4321_, v_one_4338_);
lean_dec(v_x_4321_);
v_isZero_4340_ = lean_nat_dec_eq(v_x_4322_, v_zero_4331_);
if (v_isZero_4340_ == 1)
{
lean_object* v___x_4341_; 
lean_dec(v_h__5_4327_);
lean_dec(v_x_4322_);
v___x_4341_ = lean_apply_3(v_h__4_4326_, v_head_4329_, v_tail_4330_, v_n_4339_);
return v___x_4341_;
}
else
{
lean_object* v_n_4342_; lean_object* v___x_4343_; 
lean_dec(v_h__4_4326_);
v_n_4342_ = lean_nat_sub(v_x_4322_, v_one_4338_);
lean_dec(v_x_4322_);
v___x_4343_ = lean_apply_4(v_h__5_4327_, v_head_4329_, v_tail_4330_, v_n_4339_, v_n_4342_);
return v___x_4343_;
}
}
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_List_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
