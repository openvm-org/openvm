// Lean compiler output
// Module: Aesop.Script.OptimizeSyntax
// Imports: public import Init public meta import Init public import Std.Data.HashSet.Basic meta import Lean.Parser.Term.Basic
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
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
uint8_t l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Option_bind(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadOption___lam__0(lean_object*, lean_object*);
lean_object* l_instFunctorOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Option_map(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_hash___override___boxed(lean_object*);
lean_object* l_WellFounded_opaqueFix_u2083___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__0_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__1 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__2(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__3(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticNext_=>_"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__0 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__0_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "next"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__1 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__1_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__2 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__2_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__3 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__5 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__6 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__7 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__8 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__8_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__9 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__9_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__10 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__10_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__11 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__5_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__12 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__12_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__7_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__8_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__9_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__13 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__13_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__14 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__14_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__15 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__1 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__1_value),LEAN_SCALAR_PTR_LITERAL(238, 151, 138, 49, 249, 18, 254, 242)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__2 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__2_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cdotTk"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__3 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__3_value),LEAN_SCALAR_PTR_LITERAL(117, 126, 44, 217, 38, 3, 69, 145)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__4 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__4_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__5 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__5_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__6 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__6_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__7 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__6_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__7_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__9 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__6_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__9_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeqBracketed"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__11 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__6_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__11_value),LEAN_SCALAR_PTR_LITERAL(142, 80, 121, 250, 245, 54, 71, 145)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "renameI"};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__13 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14_value_aux_0),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14_value_aux_1),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__6_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14_value_aux_2),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__13_value),LEAN_SCALAR_PTR_LITERAL(20, 41, 101, 89, 107, 117, 242, 244)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__15 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__15_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__16 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__16_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__17 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__17_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__18 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__18_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadOption___lam__3___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__19 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__19_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instFunctorOption___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__20 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__20_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_map, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__21 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__21_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__21_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__20_value)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__22 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__22_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__22_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__16_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__17_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__18_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__19_value)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__23 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__23_value;
static const lean_closure_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_bind, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__24 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__24_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__23_value),((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__24_value)}};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__25 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__25_value;
static const lean_string_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__26 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__26_value;
static const lean_ctor_object lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___boxed__const__1 = (const lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__6(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "rename_i"};
static const lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7___closed__0 = (const lean_object*)&lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7(lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__9(lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__8(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__0___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__3, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__2, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_hash___override___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__5;
static lean_once_cell_t lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__0(lean_object* v_toPure_1_, lean_object* v_____do__lift_2_){
_start:
{
uint8_t v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_3_ = 0;
v___x_4_ = l_Lean_SourceInfo_fromRef(v_____do__lift_2_, v___x_3_);
v___x_5_ = lean_apply_2(v_toPure_1_, lean_box(0), v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__0___boxed(lean_object* v_toPure_6_, lean_object* v_____do__lift_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__0(v_toPure_6_, v_____do__lift_7_);
lean_dec(v_____do__lift_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1(lean_object* v___x_13_, lean_object* v___x_14_, lean_object* v_x_15_){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; uint8_t v___x_18_; 
v___x_16_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__0));
v___x_17_ = l_Lean_Name_mkStr2(v___x_13_, v___x_16_);
lean_inc(v_x_15_);
v___x_18_ = l_Lean_Syntax_isOfKind(v_x_15_, v___x_17_);
lean_dec(v___x_17_);
if (v___x_18_ == 0)
{
lean_object* v___x_19_; 
lean_dec(v_x_15_);
v___x_19_ = lean_box(0);
return v___x_19_;
}
else
{
lean_object* v_ns_20_; lean_object* v___x_21_; uint8_t v___x_22_; 
v_ns_20_ = l_Lean_Syntax_getArg(v_x_15_, v___x_14_);
lean_dec(v_x_15_);
v___x_21_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__2));
lean_inc(v_ns_20_);
v___x_22_ = l_Lean_Syntax_isOfKind(v_ns_20_, v___x_21_);
if (v___x_22_ == 0)
{
lean_object* v___x_23_; 
lean_dec(v_ns_20_);
v___x_23_ = lean_box(0);
return v___x_23_;
}
else
{
lean_object* v___x_24_; 
v___x_24_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_24_, 0, v_ns_20_);
return v___x_24_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___boxed(lean_object* v___x_25_, lean_object* v___x_26_, lean_object* v_x_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1(v___x_25_, v___x_26_, v_x_27_);
lean_dec(v___x_26_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__2(uint8_t v___x_29_, lean_object* v_toPure_30_, lean_object* v_____do__lift_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = l_Lean_SourceInfo_fromRef(v_____do__lift_31_, v___x_29_);
v___x_33_ = lean_apply_2(v_toPure_30_, lean_box(0), v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__2___boxed(lean_object* v___x_34_, lean_object* v_toPure_35_, lean_object* v_____do__lift_36_){
_start:
{
uint8_t v___x_4968__boxed_37_; lean_object* v_res_38_; 
v___x_4968__boxed_37_ = lean_unbox(v___x_34_);
v_res_38_ = lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__2(v___x_4968__boxed_37_, v_toPure_35_, v_____do__lift_36_);
lean_dec(v_____do__lift_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__3(lean_object* v___x_39_, lean_object* v_info_40_, lean_object* v_x_41_){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__0));
v___x_43_ = l_Lean_Name_mkStr2(v___x_39_, v___x_42_);
v___x_44_ = l_Lean_Syntax_node1(v_info_40_, v___x_43_, v_x_41_);
return v___x_44_;
}
}
static lean_object* _init_lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4(void){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = l_Array_mkArray0(lean_box(0));
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4(lean_object* v___x_71_, lean_object* v___x_72_, lean_object* v___x_73_, lean_object* v_info_74_, lean_object* v_val_75_, lean_object* v___f_76_, size_t v___x_77_, lean_object* v_a_78_, lean_object* v___x_79_, lean_object* v___x_80_, lean_object* v___x_81_, lean_object* v___x_82_, lean_object* v___x_83_, lean_object* v_toPure_84_, lean_object* v_quotCtx_85_){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; size_t v_sz_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_86_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__0));
v___x_87_ = l_Lean_Name_mkStr4(v___x_71_, v___x_72_, v___x_73_, v___x_86_);
v___x_88_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__1));
lean_inc_n(v_info_74_, 6);
v___x_89_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_89_, 0, v_info_74_);
lean_ctor_set(v___x_89_, 1, v___x_88_);
v___x_90_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__3));
v___x_91_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4, &lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4);
v___x_92_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__14));
v_sz_93_ = lean_array_size(v_val_75_);
v___x_94_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_92_, v___f_76_, v_sz_93_, v___x_77_, v_val_75_);
v___x_95_ = l_Array_append___redArg(v___x_91_, v___x_94_);
lean_dec(v___x_94_);
v___x_96_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_96_, 0, v_info_74_);
lean_ctor_set(v___x_96_, 1, v___x_90_);
lean_ctor_set(v___x_96_, 2, v___x_95_);
v___x_97_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__15));
v___x_98_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_98_, 0, v_info_74_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
v___x_99_ = l_Array_toSubarray___redArg(v_a_78_, v___x_79_, v___x_80_);
v___x_100_ = l_Subarray_copy___redArg(v___x_99_);
v___x_101_ = l_Lean_Syntax_SepArray_ofElems(v___x_81_, v___x_100_);
lean_dec_ref(v___x_100_);
v___x_102_ = l_Array_append___redArg(v___x_91_, v___x_101_);
lean_dec_ref(v___x_101_);
v___x_103_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_103_, 0, v_info_74_);
lean_ctor_set(v___x_103_, 1, v___x_90_);
lean_ctor_set(v___x_103_, 2, v___x_102_);
v___x_104_ = l_Lean_Syntax_node1(v_info_74_, v___x_82_, v___x_103_);
v___x_105_ = l_Lean_Syntax_node1(v_info_74_, v___x_83_, v___x_104_);
v___x_106_ = l_Lean_Syntax_node4(v_info_74_, v___x_87_, v___x_89_, v___x_96_, v___x_98_, v___x_105_);
v___x_107_ = lean_apply_2(v_toPure_84_, lean_box(0), v___x_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___boxed(lean_object* v___x_108_, lean_object* v___x_109_, lean_object* v___x_110_, lean_object* v_info_111_, lean_object* v_val_112_, lean_object* v___f_113_, lean_object* v___x_114_, lean_object* v_a_115_, lean_object* v___x_116_, lean_object* v___x_117_, lean_object* v___x_118_, lean_object* v___x_119_, lean_object* v___x_120_, lean_object* v_toPure_121_, lean_object* v_quotCtx_122_){
_start:
{
size_t v___x_5049__boxed_123_; lean_object* v_res_124_; 
v___x_5049__boxed_123_ = lean_unbox_usize(v___x_114_);
lean_dec(v___x_114_);
v_res_124_ = lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4(v___x_108_, v___x_109_, v___x_110_, v_info_111_, v_val_112_, v___f_113_, v___x_5049__boxed_123_, v_a_115_, v___x_116_, v___x_117_, v___x_118_, v___x_119_, v___x_120_, v_toPure_121_, v_quotCtx_122_);
lean_dec(v_quotCtx_122_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__5(lean_object* v_toBind_125_, lean_object* v_getContext_126_, lean_object* v___f_127_, lean_object* v_scp_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_apply_4(v_toBind_125_, lean_box(0), lean_box(0), v_getContext_126_, v___f_127_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__5___boxed(lean_object* v_toBind_130_, lean_object* v_getContext_131_, lean_object* v___f_132_, lean_object* v_scp_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__5(v_toBind_130_, v_getContext_131_, v___f_132_, v_scp_133_);
lean_dec(v_scp_133_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__6(lean_object* v___x_135_, lean_object* v___x_136_, lean_object* v___x_137_, lean_object* v_val_138_, size_t v___x_139_, lean_object* v_a_140_, lean_object* v___x_141_, lean_object* v___x_142_, lean_object* v___x_143_, lean_object* v___x_144_, lean_object* v___x_145_, lean_object* v_toPure_146_, lean_object* v_toBind_147_, lean_object* v_getContext_148_, lean_object* v_getCurrMacroScope_149_, lean_object* v_info_150_){
_start:
{
lean_object* v___f_151_; lean_object* v___x_152_; lean_object* v___f_153_; lean_object* v___f_154_; lean_object* v___x_155_; 
lean_inc(v_info_150_);
lean_inc_ref(v___x_135_);
v___f_151_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__3), 3, 2);
lean_closure_set(v___f_151_, 0, v___x_135_);
lean_closure_set(v___f_151_, 1, v_info_150_);
v___x_152_ = lean_box_usize(v___x_139_);
v___f_153_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___boxed), 15, 14);
lean_closure_set(v___f_153_, 0, v___x_135_);
lean_closure_set(v___f_153_, 1, v___x_136_);
lean_closure_set(v___f_153_, 2, v___x_137_);
lean_closure_set(v___f_153_, 3, v_info_150_);
lean_closure_set(v___f_153_, 4, v_val_138_);
lean_closure_set(v___f_153_, 5, v___f_151_);
lean_closure_set(v___f_153_, 6, v___x_152_);
lean_closure_set(v___f_153_, 7, v_a_140_);
lean_closure_set(v___f_153_, 8, v___x_141_);
lean_closure_set(v___f_153_, 9, v___x_142_);
lean_closure_set(v___f_153_, 10, v___x_143_);
lean_closure_set(v___f_153_, 11, v___x_144_);
lean_closure_set(v___f_153_, 12, v___x_145_);
lean_closure_set(v___f_153_, 13, v_toPure_146_);
lean_inc(v_toBind_147_);
v___f_154_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__5___boxed), 4, 3);
lean_closure_set(v___f_154_, 0, v_toBind_147_);
lean_closure_set(v___f_154_, 1, v_getContext_148_);
lean_closure_set(v___f_154_, 2, v___f_153_);
v___x_155_ = lean_apply_4(v_toBind_147_, lean_box(0), lean_box(0), v_getCurrMacroScope_149_, v___f_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__6___boxed(lean_object* v___x_156_, lean_object* v___x_157_, lean_object* v___x_158_, lean_object* v_val_159_, lean_object* v___x_160_, lean_object* v_a_161_, lean_object* v___x_162_, lean_object* v___x_163_, lean_object* v___x_164_, lean_object* v___x_165_, lean_object* v___x_166_, lean_object* v_toPure_167_, lean_object* v_toBind_168_, lean_object* v_getContext_169_, lean_object* v_getCurrMacroScope_170_, lean_object* v_info_171_){
_start:
{
size_t v___x_5165__boxed_172_; lean_object* v_res_173_; 
v___x_5165__boxed_172_ = lean_unbox_usize(v___x_160_);
lean_dec(v___x_160_);
v_res_173_ = lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__6(v___x_156_, v___x_157_, v___x_158_, v_val_159_, v___x_5165__boxed_172_, v_a_161_, v___x_162_, v___x_163_, v___x_164_, v___x_165_, v___x_166_, v_toPure_167_, v_toBind_168_, v_getContext_169_, v_getCurrMacroScope_170_, v_info_171_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12(lean_object* v_info_234_, lean_object* v_kind_235_, lean_object* v_toPure_236_, lean_object* v_inst_237_, lean_object* v_toBind_238_, lean_object* v___f_239_, lean_object* v_____do__lift_240_){
_start:
{
lean_object* v_stx_241_; lean_object* v___x_242_; lean_object* v___x_243_; uint8_t v___x_244_; 
v_stx_241_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_stx_241_, 0, v_info_234_);
lean_ctor_set(v_stx_241_, 1, v_kind_235_);
lean_ctor_set(v_stx_241_, 2, v_____do__lift_240_);
v___x_242_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0));
v___x_243_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__2));
lean_inc_ref(v_stx_241_);
v___x_244_ = l_Lean_Syntax_isOfKind(v_stx_241_, v___x_243_);
if (v___x_244_ == 0)
{
lean_object* v___x_245_; 
lean_dec(v___f_239_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_245_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_245_;
}
else
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; uint8_t v___x_249_; 
v___x_246_ = lean_unsigned_to_nat(0u);
v___x_247_ = l_Lean_Syntax_getArg(v_stx_241_, v___x_246_);
v___x_248_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__4));
v___x_249_ = l_Lean_Syntax_isOfKind(v___x_247_, v___x_248_);
if (v___x_249_ == 0)
{
lean_object* v___x_250_; 
lean_dec(v___f_239_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_250_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_250_;
}
else
{
lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; uint8_t v___x_256_; 
v___x_251_ = lean_unsigned_to_nat(1u);
v___x_252_ = l_Lean_Syntax_getArg(v_stx_241_, v___x_251_);
v___x_253_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__5));
v___x_254_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__6));
v___x_255_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8));
lean_inc(v___x_252_);
v___x_256_ = l_Lean_Syntax_isOfKind(v___x_252_, v___x_255_);
if (v___x_256_ == 0)
{
lean_object* v___x_257_; 
lean_dec(v___x_252_);
lean_dec(v___f_239_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_257_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_257_;
}
else
{
lean_object* v___x_258_; lean_object* v___x_259_; uint8_t v___x_260_; 
v___x_258_ = l_Lean_Syntax_getArg(v___x_252_, v___x_246_);
lean_dec(v___x_252_);
v___x_259_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10));
lean_inc(v___x_258_);
v___x_260_ = l_Lean_Syntax_isOfKind(v___x_258_, v___x_259_);
if (v___x_260_ == 0)
{
lean_object* v___x_261_; uint8_t v___x_262_; 
lean_dec(v___f_239_);
v___x_261_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__12));
lean_inc(v___x_258_);
v___x_262_ = l_Lean_Syntax_isOfKind(v___x_258_, v___x_261_);
if (v___x_262_ == 0)
{
lean_object* v___x_263_; 
lean_dec(v___x_258_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_263_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_263_;
}
else
{
lean_object* v___x_264_; lean_object* v_tacs_265_; lean_object* v_a_266_; lean_object* v___x_267_; uint8_t v___x_268_; 
v___x_264_ = l_Lean_Syntax_getArg(v___x_258_, v___x_251_);
lean_dec(v___x_258_);
v_tacs_265_ = l_Lean_Syntax_getArgs(v___x_264_);
lean_dec(v___x_264_);
v_a_266_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_tacs_265_);
lean_dec_ref(v_tacs_265_);
v___x_267_ = lean_array_get_size(v_a_266_);
v___x_268_ = lean_nat_dec_lt(v___x_246_, v___x_267_);
if (v___x_268_ == 0)
{
lean_object* v___x_269_; 
lean_dec_ref(v_a_266_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_269_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_269_;
}
else
{
lean_object* v___x_270_; lean_object* v___x_271_; uint8_t v___x_272_; 
v___x_270_ = lean_array_fget(v_a_266_, v___x_246_);
v___x_271_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14));
lean_inc(v___x_270_);
v___x_272_ = l_Lean_Syntax_isOfKind(v___x_270_, v___x_271_);
if (v___x_272_ == 0)
{
lean_object* v___x_273_; 
lean_dec(v___x_270_);
lean_dec_ref(v_a_266_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_273_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_273_;
}
else
{
lean_object* v___f_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; size_t v_sz_278_; size_t v___x_279_; lean_object* v___x_280_; 
v___f_274_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__15));
v___x_275_ = l_Lean_Syntax_getArg(v___x_270_, v___x_251_);
lean_dec(v___x_270_);
v___x_276_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__25));
v___x_277_ = l_Lean_Syntax_getArgs(v___x_275_);
lean_dec(v___x_275_);
v_sz_278_ = lean_array_size(v___x_277_);
v___x_279_ = ((size_t)0ULL);
v___x_280_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_276_, v___f_274_, v_sz_278_, v___x_279_, v___x_277_);
if (lean_obj_tag(v___x_280_) == 0)
{
lean_object* v___x_281_; 
lean_dec_ref(v_a_266_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_281_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_281_;
}
else
{
lean_object* v_toMonadRef_282_; lean_object* v_val_283_; lean_object* v_getCurrMacroScope_284_; lean_object* v_getContext_285_; lean_object* v_getRef_286_; lean_object* v___x_287_; lean_object* v___f_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___f_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
lean_dec_ref_known(v_stx_241_, 3);
v_toMonadRef_282_ = lean_ctor_get(v_inst_237_, 0);
lean_inc_ref(v_toMonadRef_282_);
v_val_283_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_val_283_);
lean_dec_ref_known(v___x_280_, 1);
v_getCurrMacroScope_284_ = lean_ctor_get(v_inst_237_, 1);
lean_inc(v_getCurrMacroScope_284_);
v_getContext_285_ = lean_ctor_get(v_inst_237_, 2);
lean_inc(v_getContext_285_);
lean_dec_ref(v_inst_237_);
v_getRef_286_ = lean_ctor_get(v_toMonadRef_282_, 0);
lean_inc(v_getRef_286_);
lean_dec_ref(v_toMonadRef_282_);
v___x_287_ = lean_box(v___x_260_);
lean_inc(v_toPure_236_);
v___f_288_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_288_, 0, v___x_287_);
lean_closure_set(v___f_288_, 1, v_toPure_236_);
v___x_289_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__26));
v___x_290_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___boxed__const__1));
lean_inc_n(v_toBind_238_, 2);
v___f_291_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__6___boxed), 16, 15);
lean_closure_set(v___f_291_, 0, v___x_242_);
lean_closure_set(v___f_291_, 1, v___x_253_);
lean_closure_set(v___f_291_, 2, v___x_254_);
lean_closure_set(v___f_291_, 3, v_val_283_);
lean_closure_set(v___f_291_, 4, v___x_290_);
lean_closure_set(v___f_291_, 5, v_a_266_);
lean_closure_set(v___f_291_, 6, v___x_251_);
lean_closure_set(v___f_291_, 7, v___x_267_);
lean_closure_set(v___f_291_, 8, v___x_289_);
lean_closure_set(v___f_291_, 9, v___x_259_);
lean_closure_set(v___f_291_, 10, v___x_255_);
lean_closure_set(v___f_291_, 11, v_toPure_236_);
lean_closure_set(v___f_291_, 12, v_toBind_238_);
lean_closure_set(v___f_291_, 13, v_getContext_285_);
lean_closure_set(v___f_291_, 14, v_getCurrMacroScope_284_);
v___x_292_ = lean_apply_4(v_toBind_238_, lean_box(0), lean_box(0), v_getRef_286_, v___f_288_);
v___x_293_ = lean_apply_4(v_toBind_238_, lean_box(0), lean_box(0), v___x_292_, v___f_291_);
return v___x_293_;
}
}
}
}
}
else
{
lean_object* v___x_294_; lean_object* v_tacs_295_; lean_object* v_a_296_; lean_object* v___x_297_; uint8_t v___x_298_; 
v___x_294_ = l_Lean_Syntax_getArg(v___x_258_, v___x_246_);
lean_dec(v___x_258_);
v_tacs_295_ = l_Lean_Syntax_getArgs(v___x_294_);
lean_dec(v___x_294_);
v_a_296_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_tacs_295_);
lean_dec_ref(v_tacs_295_);
v___x_297_ = lean_array_get_size(v_a_296_);
v___x_298_ = lean_nat_dec_lt(v___x_246_, v___x_297_);
if (v___x_298_ == 0)
{
lean_object* v___x_299_; 
lean_dec_ref(v_a_296_);
lean_dec(v___f_239_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_299_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_299_;
}
else
{
lean_object* v___x_300_; lean_object* v___x_301_; uint8_t v___x_302_; 
v___x_300_ = lean_array_fget(v_a_296_, v___x_246_);
v___x_301_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14));
lean_inc(v___x_300_);
v___x_302_ = l_Lean_Syntax_isOfKind(v___x_300_, v___x_301_);
if (v___x_302_ == 0)
{
lean_object* v___x_303_; 
lean_dec(v___x_300_);
lean_dec_ref(v_a_296_);
lean_dec(v___f_239_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_303_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_303_;
}
else
{
lean_object* v___f_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; size_t v_sz_308_; size_t v___x_309_; lean_object* v___x_310_; 
v___f_304_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__15));
v___x_305_ = l_Lean_Syntax_getArg(v___x_300_, v___x_251_);
lean_dec(v___x_300_);
v___x_306_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__25));
v___x_307_ = l_Lean_Syntax_getArgs(v___x_305_);
lean_dec(v___x_305_);
v_sz_308_ = lean_array_size(v___x_307_);
v___x_309_ = ((size_t)0ULL);
v___x_310_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_306_, v___f_304_, v_sz_308_, v___x_309_, v___x_307_);
if (lean_obj_tag(v___x_310_) == 0)
{
lean_object* v___x_311_; 
lean_dec_ref(v_a_296_);
lean_dec(v___f_239_);
lean_dec(v_toBind_238_);
lean_dec_ref(v_inst_237_);
v___x_311_ = lean_apply_2(v_toPure_236_, lean_box(0), v_stx_241_);
return v___x_311_;
}
else
{
lean_object* v_toMonadRef_312_; lean_object* v_val_313_; lean_object* v_getCurrMacroScope_314_; lean_object* v_getContext_315_; lean_object* v_getRef_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___f_319_; lean_object* v___x_320_; lean_object* v___x_321_; 
lean_dec_ref_known(v_stx_241_, 3);
v_toMonadRef_312_ = lean_ctor_get(v_inst_237_, 0);
lean_inc_ref(v_toMonadRef_312_);
v_val_313_ = lean_ctor_get(v___x_310_, 0);
lean_inc(v_val_313_);
lean_dec_ref_known(v___x_310_, 1);
v_getCurrMacroScope_314_ = lean_ctor_get(v_inst_237_, 1);
lean_inc(v_getCurrMacroScope_314_);
v_getContext_315_ = lean_ctor_get(v_inst_237_, 2);
lean_inc(v_getContext_315_);
lean_dec_ref(v_inst_237_);
v_getRef_316_ = lean_ctor_get(v_toMonadRef_312_, 0);
lean_inc(v_getRef_316_);
lean_dec_ref(v_toMonadRef_312_);
v___x_317_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__26));
v___x_318_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___boxed__const__1));
lean_inc_n(v_toBind_238_, 2);
v___f_319_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__6___boxed), 16, 15);
lean_closure_set(v___f_319_, 0, v___x_242_);
lean_closure_set(v___f_319_, 1, v___x_253_);
lean_closure_set(v___f_319_, 2, v___x_254_);
lean_closure_set(v___f_319_, 3, v_val_313_);
lean_closure_set(v___f_319_, 4, v___x_318_);
lean_closure_set(v___f_319_, 5, v_a_296_);
lean_closure_set(v___f_319_, 6, v___x_251_);
lean_closure_set(v___f_319_, 7, v___x_297_);
lean_closure_set(v___f_319_, 8, v___x_317_);
lean_closure_set(v___f_319_, 9, v___x_259_);
lean_closure_set(v___f_319_, 10, v___x_255_);
lean_closure_set(v___f_319_, 11, v_toPure_236_);
lean_closure_set(v___f_319_, 12, v_toBind_238_);
lean_closure_set(v___f_319_, 13, v_getContext_315_);
lean_closure_set(v___f_319_, 14, v_getCurrMacroScope_314_);
v___x_320_ = lean_apply_4(v_toBind_238_, lean_box(0), lean_box(0), v_getRef_316_, v___f_239_);
v___x_321_ = lean_apply_4(v_toBind_238_, lean_box(0), lean_box(0), v___x_320_, v___f_319_);
return v___x_321_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI___redArg(lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_x_324_){
_start:
{
if (lean_obj_tag(v_x_324_) == 1)
{
lean_object* v_toApplicative_325_; lean_object* v_toBind_326_; lean_object* v_toPure_327_; lean_object* v_info_328_; lean_object* v_kind_329_; lean_object* v_args_330_; lean_object* v___f_331_; lean_object* v___f_332_; lean_object* v___x_333_; size_t v_sz_334_; size_t v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; 
v_toApplicative_325_ = lean_ctor_get(v_inst_322_, 0);
v_toBind_326_ = lean_ctor_get(v_inst_322_, 1);
lean_inc_n(v_toBind_326_, 2);
v_toPure_327_ = lean_ctor_get(v_toApplicative_325_, 1);
v_info_328_ = lean_ctor_get(v_x_324_, 0);
lean_inc(v_info_328_);
v_kind_329_ = lean_ctor_get(v_x_324_, 1);
lean_inc(v_kind_329_);
v_args_330_ = lean_ctor_get(v_x_324_, 2);
lean_inc_ref(v_args_330_);
lean_dec_ref_known(v_x_324_, 3);
lean_inc_n(v_toPure_327_, 2);
v___f_331_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_331_, 0, v_toPure_327_);
lean_inc_ref(v_inst_323_);
v___f_332_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12), 7, 6);
lean_closure_set(v___f_332_, 0, v_info_328_);
lean_closure_set(v___f_332_, 1, v_kind_329_);
lean_closure_set(v___f_332_, 2, v_toPure_327_);
lean_closure_set(v___f_332_, 3, v_inst_323_);
lean_closure_set(v___f_332_, 4, v_toBind_326_);
lean_closure_set(v___f_332_, 5, v___f_331_);
lean_inc_ref(v_inst_322_);
v___x_333_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg), 3, 2);
lean_closure_set(v___x_333_, 0, v_inst_322_);
lean_closure_set(v___x_333_, 1, v_inst_323_);
v_sz_334_ = lean_array_size(v_args_330_);
v___x_335_ = ((size_t)0ULL);
v___x_336_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v_inst_322_, v___x_333_, v_sz_334_, v___x_335_, v_args_330_);
v___x_337_ = lean_apply_4(v_toBind_326_, lean_box(0), lean_box(0), v___x_336_, v___f_332_);
return v___x_337_;
}
else
{
lean_object* v_toApplicative_338_; lean_object* v_toPure_339_; lean_object* v___x_340_; 
v_toApplicative_338_ = lean_ctor_get(v_inst_322_, 0);
lean_inc_ref(v_toApplicative_338_);
lean_dec_ref(v_inst_323_);
lean_dec_ref(v_inst_322_);
v_toPure_339_ = lean_ctor_get(v_toApplicative_338_, 1);
lean_inc(v_toPure_339_);
lean_dec_ref(v_toApplicative_338_);
v___x_340_ = lean_apply_2(v_toPure_339_, lean_box(0), v_x_324_);
return v___x_340_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeFocusRenameI(lean_object* v_m_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_x_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lp_aesop_Aesop_optimizeFocusRenameI___redArg(v_inst_342_, v_inst_343_, v_x_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3_spec__4___redArg(lean_object* v_x_346_, lean_object* v_x_347_){
_start:
{
if (lean_obj_tag(v_x_347_) == 0)
{
return v_x_346_;
}
else
{
lean_object* v_key_348_; lean_object* v_value_349_; lean_object* v_tail_350_; lean_object* v___x_352_; uint8_t v_isShared_353_; uint8_t v_isSharedCheck_376_; 
v_key_348_ = lean_ctor_get(v_x_347_, 0);
v_value_349_ = lean_ctor_get(v_x_347_, 1);
v_tail_350_ = lean_ctor_get(v_x_347_, 2);
v_isSharedCheck_376_ = !lean_is_exclusive(v_x_347_);
if (v_isSharedCheck_376_ == 0)
{
v___x_352_ = v_x_347_;
v_isShared_353_ = v_isSharedCheck_376_;
goto v_resetjp_351_;
}
else
{
lean_inc(v_tail_350_);
lean_inc(v_value_349_);
lean_inc(v_key_348_);
lean_dec(v_x_347_);
v___x_352_ = lean_box(0);
v_isShared_353_ = v_isSharedCheck_376_;
goto v_resetjp_351_;
}
v_resetjp_351_:
{
lean_object* v___x_354_; uint64_t v___y_356_; 
v___x_354_ = lean_array_get_size(v_x_346_);
if (lean_obj_tag(v_key_348_) == 0)
{
uint64_t v___x_374_; 
v___x_374_ = 1723ULL;
v___y_356_ = v___x_374_;
goto v___jp_355_;
}
else
{
uint64_t v_hash_375_; 
v_hash_375_ = lean_ctor_get_uint64(v_key_348_, sizeof(void*)*2);
v___y_356_ = v_hash_375_;
goto v___jp_355_;
}
v___jp_355_:
{
uint64_t v___x_357_; uint64_t v___x_358_; uint64_t v_fold_359_; uint64_t v___x_360_; uint64_t v___x_361_; uint64_t v___x_362_; size_t v___x_363_; size_t v___x_364_; size_t v___x_365_; size_t v___x_366_; size_t v___x_367_; lean_object* v___x_368_; lean_object* v___x_370_; 
v___x_357_ = 32ULL;
v___x_358_ = lean_uint64_shift_right(v___y_356_, v___x_357_);
v_fold_359_ = lean_uint64_xor(v___y_356_, v___x_358_);
v___x_360_ = 16ULL;
v___x_361_ = lean_uint64_shift_right(v_fold_359_, v___x_360_);
v___x_362_ = lean_uint64_xor(v_fold_359_, v___x_361_);
v___x_363_ = lean_uint64_to_usize(v___x_362_);
v___x_364_ = lean_usize_of_nat(v___x_354_);
v___x_365_ = ((size_t)1ULL);
v___x_366_ = lean_usize_sub(v___x_364_, v___x_365_);
v___x_367_ = lean_usize_land(v___x_363_, v___x_366_);
v___x_368_ = lean_array_uget_borrowed(v_x_346_, v___x_367_);
lean_inc(v___x_368_);
if (v_isShared_353_ == 0)
{
lean_ctor_set(v___x_352_, 2, v___x_368_);
v___x_370_ = v___x_352_;
goto v_reusejp_369_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v_key_348_);
lean_ctor_set(v_reuseFailAlloc_373_, 1, v_value_349_);
lean_ctor_set(v_reuseFailAlloc_373_, 2, v___x_368_);
v___x_370_ = v_reuseFailAlloc_373_;
goto v_reusejp_369_;
}
v_reusejp_369_:
{
lean_object* v___x_371_; 
v___x_371_ = lean_array_uset(v_x_346_, v___x_367_, v___x_370_);
v_x_346_ = v___x_371_;
v_x_347_ = v_tail_350_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3___redArg(lean_object* v_i_377_, lean_object* v_source_378_, lean_object* v_target_379_){
_start:
{
lean_object* v___x_380_; uint8_t v___x_381_; 
v___x_380_ = lean_array_get_size(v_source_378_);
v___x_381_ = lean_nat_dec_lt(v_i_377_, v___x_380_);
if (v___x_381_ == 0)
{
lean_dec_ref(v_source_378_);
lean_dec(v_i_377_);
return v_target_379_;
}
else
{
lean_object* v_es_382_; lean_object* v___x_383_; lean_object* v_source_384_; lean_object* v_target_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
v_es_382_ = lean_array_fget(v_source_378_, v_i_377_);
v___x_383_ = lean_box(0);
v_source_384_ = lean_array_fset(v_source_378_, v_i_377_, v___x_383_);
v_target_385_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3_spec__4___redArg(v_target_379_, v_es_382_);
v___x_386_ = lean_unsigned_to_nat(1u);
v___x_387_ = lean_nat_add(v_i_377_, v___x_386_);
lean_dec(v_i_377_);
v_i_377_ = v___x_387_;
v_source_378_ = v_source_384_;
v_target_379_ = v_target_385_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2___redArg(lean_object* v_data_389_){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v_nbuckets_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_390_ = lean_array_get_size(v_data_389_);
v___x_391_ = lean_unsigned_to_nat(2u);
v_nbuckets_392_ = lean_nat_mul(v___x_390_, v___x_391_);
v___x_393_ = lean_unsigned_to_nat(0u);
v___x_394_ = lean_box(0);
v___x_395_ = lean_mk_array(v_nbuckets_392_, v___x_394_);
v___x_396_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3___redArg(v___x_393_, v_data_389_, v___x_395_);
return v___x_396_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___redArg(lean_object* v_a_397_, lean_object* v_x_398_){
_start:
{
if (lean_obj_tag(v_x_398_) == 0)
{
uint8_t v___x_399_; 
v___x_399_ = 0;
return v___x_399_;
}
else
{
lean_object* v_key_400_; lean_object* v_tail_401_; uint8_t v___x_402_; 
v_key_400_ = lean_ctor_get(v_x_398_, 0);
v_tail_401_ = lean_ctor_get(v_x_398_, 2);
v___x_402_ = lean_name_eq(v_key_400_, v_a_397_);
if (v___x_402_ == 0)
{
v_x_398_ = v_tail_401_;
goto _start;
}
else
{
return v___x_402_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___redArg___boxed(lean_object* v_a_404_, lean_object* v_x_405_){
_start:
{
uint8_t v_res_406_; lean_object* v_r_407_; 
v_res_406_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___redArg(v_a_404_, v_x_405_);
lean_dec(v_x_405_);
lean_dec(v_a_404_);
v_r_407_ = lean_box(v_res_406_);
return v_r_407_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1___redArg(lean_object* v_m_408_, lean_object* v_a_409_, lean_object* v_b_410_){
_start:
{
lean_object* v_size_411_; lean_object* v_buckets_412_; lean_object* v___x_413_; uint64_t v___y_415_; 
v_size_411_ = lean_ctor_get(v_m_408_, 0);
v_buckets_412_ = lean_ctor_get(v_m_408_, 1);
v___x_413_ = lean_array_get_size(v_buckets_412_);
if (lean_obj_tag(v_a_409_) == 0)
{
uint64_t v___x_452_; 
v___x_452_ = 1723ULL;
v___y_415_ = v___x_452_;
goto v___jp_414_;
}
else
{
uint64_t v_hash_453_; 
v_hash_453_ = lean_ctor_get_uint64(v_a_409_, sizeof(void*)*2);
v___y_415_ = v_hash_453_;
goto v___jp_414_;
}
v___jp_414_:
{
uint64_t v___x_416_; uint64_t v___x_417_; uint64_t v_fold_418_; uint64_t v___x_419_; uint64_t v___x_420_; uint64_t v___x_421_; size_t v___x_422_; size_t v___x_423_; size_t v___x_424_; size_t v___x_425_; size_t v___x_426_; lean_object* v_bkt_427_; uint8_t v___x_428_; 
v___x_416_ = 32ULL;
v___x_417_ = lean_uint64_shift_right(v___y_415_, v___x_416_);
v_fold_418_ = lean_uint64_xor(v___y_415_, v___x_417_);
v___x_419_ = 16ULL;
v___x_420_ = lean_uint64_shift_right(v_fold_418_, v___x_419_);
v___x_421_ = lean_uint64_xor(v_fold_418_, v___x_420_);
v___x_422_ = lean_uint64_to_usize(v___x_421_);
v___x_423_ = lean_usize_of_nat(v___x_413_);
v___x_424_ = ((size_t)1ULL);
v___x_425_ = lean_usize_sub(v___x_423_, v___x_424_);
v___x_426_ = lean_usize_land(v___x_422_, v___x_425_);
v_bkt_427_ = lean_array_uget_borrowed(v_buckets_412_, v___x_426_);
v___x_428_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___redArg(v_a_409_, v_bkt_427_);
if (v___x_428_ == 0)
{
lean_object* v___x_430_; uint8_t v_isShared_431_; uint8_t v_isSharedCheck_449_; 
lean_inc_ref(v_buckets_412_);
lean_inc(v_size_411_);
v_isSharedCheck_449_ = !lean_is_exclusive(v_m_408_);
if (v_isSharedCheck_449_ == 0)
{
lean_object* v_unused_450_; lean_object* v_unused_451_; 
v_unused_450_ = lean_ctor_get(v_m_408_, 1);
lean_dec(v_unused_450_);
v_unused_451_ = lean_ctor_get(v_m_408_, 0);
lean_dec(v_unused_451_);
v___x_430_ = v_m_408_;
v_isShared_431_ = v_isSharedCheck_449_;
goto v_resetjp_429_;
}
else
{
lean_dec(v_m_408_);
v___x_430_ = lean_box(0);
v_isShared_431_ = v_isSharedCheck_449_;
goto v_resetjp_429_;
}
v_resetjp_429_:
{
lean_object* v___x_432_; lean_object* v_size_x27_433_; lean_object* v___x_434_; lean_object* v_buckets_x27_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; uint8_t v___x_441_; 
v___x_432_ = lean_unsigned_to_nat(1u);
v_size_x27_433_ = lean_nat_add(v_size_411_, v___x_432_);
lean_dec(v_size_411_);
lean_inc(v_bkt_427_);
v___x_434_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_434_, 0, v_a_409_);
lean_ctor_set(v___x_434_, 1, v_b_410_);
lean_ctor_set(v___x_434_, 2, v_bkt_427_);
v_buckets_x27_435_ = lean_array_uset(v_buckets_412_, v___x_426_, v___x_434_);
v___x_436_ = lean_unsigned_to_nat(4u);
v___x_437_ = lean_nat_mul(v_size_x27_433_, v___x_436_);
v___x_438_ = lean_unsigned_to_nat(3u);
v___x_439_ = lean_nat_div(v___x_437_, v___x_438_);
lean_dec(v___x_437_);
v___x_440_ = lean_array_get_size(v_buckets_x27_435_);
v___x_441_ = lean_nat_dec_le(v___x_439_, v___x_440_);
lean_dec(v___x_439_);
if (v___x_441_ == 0)
{
lean_object* v_val_442_; lean_object* v___x_444_; 
v_val_442_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2___redArg(v_buckets_x27_435_);
if (v_isShared_431_ == 0)
{
lean_ctor_set(v___x_430_, 1, v_val_442_);
lean_ctor_set(v___x_430_, 0, v_size_x27_433_);
v___x_444_ = v___x_430_;
goto v_reusejp_443_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v_size_x27_433_);
lean_ctor_set(v_reuseFailAlloc_445_, 1, v_val_442_);
v___x_444_ = v_reuseFailAlloc_445_;
goto v_reusejp_443_;
}
v_reusejp_443_:
{
return v___x_444_;
}
}
else
{
lean_object* v___x_447_; 
if (v_isShared_431_ == 0)
{
lean_ctor_set(v___x_430_, 1, v_buckets_x27_435_);
lean_ctor_set(v___x_430_, 0, v_size_x27_433_);
v___x_447_ = v___x_430_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v_size_x27_433_);
lean_ctor_set(v_reuseFailAlloc_448_, 1, v_buckets_x27_435_);
v___x_447_ = v_reuseFailAlloc_448_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
return v___x_447_;
}
}
}
}
else
{
lean_dec(v_b_410_);
lean_dec(v_a_409_);
return v_m_408_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents(lean_object* v_acc_454_, lean_object* v_x_455_){
_start:
{
switch(lean_obj_tag(v_x_455_))
{
case 1:
{
lean_object* v_args_456_; lean_object* v___x_457_; lean_object* v___x_458_; uint8_t v___x_459_; 
v_args_456_ = lean_ctor_get(v_x_455_, 2);
lean_inc_ref(v_args_456_);
lean_dec_ref_known(v_x_455_, 3);
v___x_457_ = lean_unsigned_to_nat(0u);
v___x_458_ = lean_array_get_size(v_args_456_);
v___x_459_ = lean_nat_dec_lt(v___x_457_, v___x_458_);
if (v___x_459_ == 0)
{
lean_dec_ref(v_args_456_);
return v_acc_454_;
}
else
{
uint8_t v___x_460_; 
v___x_460_ = lean_nat_dec_le(v___x_458_, v___x_458_);
if (v___x_460_ == 0)
{
if (v___x_459_ == 0)
{
lean_dec_ref(v_args_456_);
return v_acc_454_;
}
else
{
size_t v___x_461_; size_t v___x_462_; lean_object* v___x_463_; 
v___x_461_ = ((size_t)0ULL);
v___x_462_ = lean_usize_of_nat(v___x_458_);
v___x_463_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__0(v_args_456_, v___x_461_, v___x_462_, v_acc_454_);
lean_dec_ref(v_args_456_);
return v___x_463_;
}
}
else
{
size_t v___x_464_; size_t v___x_465_; lean_object* v___x_466_; 
v___x_464_ = ((size_t)0ULL);
v___x_465_ = lean_usize_of_nat(v___x_458_);
v___x_466_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__0(v_args_456_, v___x_464_, v___x_465_, v_acc_454_);
lean_dec_ref(v_args_456_);
return v___x_466_;
}
}
}
case 3:
{
lean_object* v_val_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v_val_467_ = lean_ctor_get(v_x_455_, 2);
lean_inc(v_val_467_);
lean_dec_ref_known(v_x_455_, 4);
v___x_468_ = lean_box(0);
v___x_469_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1___redArg(v_acc_454_, v_val_467_, v___x_468_);
return v___x_469_;
}
default: 
{
lean_dec(v_x_455_);
return v_acc_454_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__0(lean_object* v_as_470_, size_t v_i_471_, size_t v_stop_472_, lean_object* v_b_473_){
_start:
{
uint8_t v___x_474_; 
v___x_474_ = lean_usize_dec_eq(v_i_471_, v_stop_472_);
if (v___x_474_ == 0)
{
lean_object* v___x_475_; lean_object* v___x_476_; size_t v___x_477_; size_t v___x_478_; 
v___x_475_ = lean_array_uget_borrowed(v_as_470_, v_i_471_);
lean_inc(v___x_475_);
v___x_476_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents(v_b_473_, v___x_475_);
v___x_477_ = ((size_t)1ULL);
v___x_478_ = lean_usize_add(v_i_471_, v___x_477_);
v_i_471_ = v___x_478_;
v_b_473_ = v___x_476_;
goto _start;
}
else
{
return v_b_473_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__0___boxed(lean_object* v_as_480_, lean_object* v_i_481_, lean_object* v_stop_482_, lean_object* v_b_483_){
_start:
{
size_t v_i_boxed_484_; size_t v_stop_boxed_485_; lean_object* v_res_486_; 
v_i_boxed_484_ = lean_unbox_usize(v_i_481_);
lean_dec(v_i_481_);
v_stop_boxed_485_ = lean_unbox_usize(v_stop_482_);
lean_dec(v_stop_482_);
v_res_486_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__0(v_as_480_, v_i_boxed_484_, v_stop_boxed_485_, v_b_483_);
lean_dec_ref(v_as_480_);
return v_res_486_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1(lean_object* v_00_u03b2_487_, lean_object* v_m_488_, lean_object* v_a_489_, lean_object* v_b_490_){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1___redArg(v_m_488_, v_a_489_, v_b_490_);
return v___x_491_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1(lean_object* v_00_u03b2_492_, lean_object* v_a_493_, lean_object* v_x_494_){
_start:
{
uint8_t v___x_495_; 
v___x_495_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___redArg(v_a_493_, v_x_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1___boxed(lean_object* v_00_u03b2_496_, lean_object* v_a_497_, lean_object* v_x_498_){
_start:
{
uint8_t v_res_499_; lean_object* v_r_500_; 
v_res_499_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__1(v_00_u03b2_496_, v_a_497_, v_x_498_);
lean_dec(v_x_498_);
lean_dec(v_a_497_);
v_r_500_ = lean_box(v_res_499_);
return v_r_500_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2(lean_object* v_00_u03b2_501_, lean_object* v_data_502_){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2___redArg(v_data_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_504_, lean_object* v_i_505_, lean_object* v_source_506_, lean_object* v_target_507_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3___redArg(v_i_505_, v_source_506_, v_target_507_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3_spec__4(lean_object* v_00_u03b2_509_, lean_object* v_x_510_, lean_object* v_x_511_){
_start:
{
lean_object* v___x_512_; 
v___x_512_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00__private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents_spec__1_spec__2_spec__3_spec__4___redArg(v_x_510_, v_x_511_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__0(lean_object* v_tacs_513_, lean_object* v_info_514_, lean_object* v_toPure_515_, lean_object* v_quotCtx_516_){
_start:
{
lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; 
v___x_517_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8));
v___x_518_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10));
v___x_519_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__3));
v___x_520_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4, &lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4);
v___x_521_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__26));
v___x_522_ = l_Lean_Syntax_SepArray_ofElems(v___x_521_, v_tacs_513_);
v___x_523_ = l_Array_append___redArg(v___x_520_, v___x_522_);
lean_dec_ref(v___x_522_);
lean_inc_n(v_info_514_, 2);
v___x_524_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_524_, 0, v_info_514_);
lean_ctor_set(v___x_524_, 1, v___x_519_);
lean_ctor_set(v___x_524_, 2, v___x_523_);
v___x_525_ = l_Lean_Syntax_node1(v_info_514_, v___x_518_, v___x_524_);
v___x_526_ = l_Lean_Syntax_node1(v_info_514_, v___x_517_, v___x_525_);
v___x_527_ = lean_apply_2(v_toPure_515_, lean_box(0), v___x_526_);
return v___x_527_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__0___boxed(lean_object* v_tacs_528_, lean_object* v_info_529_, lean_object* v_toPure_530_, lean_object* v_quotCtx_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__0(v_tacs_528_, v_info_529_, v_toPure_530_, v_quotCtx_531_);
lean_dec(v_quotCtx_531_);
lean_dec_ref(v_tacs_528_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__1(lean_object* v_toBind_533_, lean_object* v_getContext_534_, lean_object* v___f_535_, lean_object* v_scp_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lean_apply_4(v_toBind_533_, lean_box(0), lean_box(0), v_getContext_534_, v___f_535_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__1___boxed(lean_object* v_toBind_538_, lean_object* v_getContext_539_, lean_object* v___f_540_, lean_object* v_scp_541_){
_start:
{
lean_object* v_res_542_; 
v_res_542_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__1(v_toBind_538_, v_getContext_539_, v___f_540_, v_scp_541_);
lean_dec(v_scp_541_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__2(lean_object* v_tacs_543_, lean_object* v_toPure_544_, lean_object* v_toBind_545_, lean_object* v_getContext_546_, lean_object* v_getCurrMacroScope_547_, lean_object* v_info_548_){
_start:
{
lean_object* v___f_549_; lean_object* v___f_550_; lean_object* v___x_551_; 
v___f_549_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_549_, 0, v_tacs_543_);
lean_closure_set(v___f_549_, 1, v_info_548_);
lean_closure_set(v___f_549_, 2, v_toPure_544_);
lean_inc(v_toBind_545_);
v___f_550_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_550_, 0, v_toBind_545_);
lean_closure_set(v___f_550_, 1, v_getContext_546_);
lean_closure_set(v___f_550_, 2, v___f_549_);
v___x_551_ = lean_apply_4(v_toBind_545_, lean_box(0), lean_box(0), v_getCurrMacroScope_547_, v___f_550_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__3(lean_object* v_toPure_552_, lean_object* v_____do__lift_553_){
_start:
{
uint8_t v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; 
v___x_554_ = 0;
v___x_555_ = l_Lean_SourceInfo_fromRef(v_____do__lift_553_, v___x_554_);
v___x_556_ = lean_apply_2(v_toPure_552_, lean_box(0), v___x_555_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__3___boxed(lean_object* v_toPure_557_, lean_object* v_____do__lift_558_){
_start:
{
lean_object* v_res_559_; 
v_res_559_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__3(v_toPure_557_, v_____do__lift_558_);
lean_dec(v_____do__lift_558_);
return v_res_559_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg(lean_object* v_inst_560_, lean_object* v_inst_561_, lean_object* v_tacs_562_){
_start:
{
lean_object* v_toMonadRef_563_; lean_object* v_toApplicative_564_; lean_object* v_toBind_565_; lean_object* v_getCurrMacroScope_566_; lean_object* v_getContext_567_; lean_object* v_getRef_568_; lean_object* v_toPure_569_; lean_object* v___f_570_; lean_object* v___f_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
v_toMonadRef_563_ = lean_ctor_get(v_inst_561_, 0);
lean_inc_ref(v_toMonadRef_563_);
v_toApplicative_564_ = lean_ctor_get(v_inst_560_, 0);
lean_inc_ref(v_toApplicative_564_);
v_toBind_565_ = lean_ctor_get(v_inst_560_, 1);
lean_inc_n(v_toBind_565_, 3);
lean_dec_ref(v_inst_560_);
v_getCurrMacroScope_566_ = lean_ctor_get(v_inst_561_, 1);
lean_inc(v_getCurrMacroScope_566_);
v_getContext_567_ = lean_ctor_get(v_inst_561_, 2);
lean_inc(v_getContext_567_);
lean_dec_ref(v_inst_561_);
v_getRef_568_ = lean_ctor_get(v_toMonadRef_563_, 0);
lean_inc(v_getRef_568_);
lean_dec_ref(v_toMonadRef_563_);
v_toPure_569_ = lean_ctor_get(v_toApplicative_564_, 1);
lean_inc_n(v_toPure_569_, 2);
lean_dec_ref(v_toApplicative_564_);
v___f_570_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__2), 6, 5);
lean_closure_set(v___f_570_, 0, v_tacs_562_);
lean_closure_set(v___f_570_, 1, v_toPure_569_);
lean_closure_set(v___f_570_, 2, v_toBind_565_);
lean_closure_set(v___f_570_, 3, v_getContext_567_);
lean_closure_set(v___f_570_, 4, v_getCurrMacroScope_566_);
v___f_571_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_571_, 0, v_toPure_569_);
v___x_572_ = lean_apply_4(v_toBind_565_, lean_box(0), lean_box(0), v_getRef_568_, v___f_571_);
v___x_573_ = lean_apply_4(v_toBind_565_, lean_box(0), lean_box(0), v___x_572_, v___f_570_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq(lean_object* v_m_574_, lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_tacs_577_){
_start:
{
lean_object* v___x_578_; 
v___x_578_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg(v_inst_575_, v_inst_576_, v_tacs_577_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__0(lean_object* v___x_579_, lean_object* v_dropUntil_580_, lean_object* v_x_581_){
_start:
{
lean_object* v___x_582_; lean_object* v___x_583_; uint8_t v___x_584_; 
v___x_582_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__0));
v___x_583_ = l_Lean_Name_mkStr2(v___x_579_, v___x_582_);
lean_inc(v_x_581_);
v___x_584_ = l_Lean_Syntax_isOfKind(v_x_581_, v___x_583_);
lean_dec(v___x_583_);
if (v___x_584_ == 0)
{
lean_object* v___x_585_; 
lean_dec(v_x_581_);
v___x_585_ = lean_box(0);
return v___x_585_;
}
else
{
lean_object* v_ns_586_; lean_object* v___x_587_; uint8_t v___x_588_; 
v_ns_586_ = l_Lean_Syntax_getArg(v_x_581_, v_dropUntil_580_);
lean_dec(v_x_581_);
v___x_587_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__2));
lean_inc(v_ns_586_);
v___x_588_ = l_Lean_Syntax_isOfKind(v_ns_586_, v___x_587_);
if (v___x_588_ == 0)
{
lean_object* v___x_589_; 
lean_dec(v_ns_586_);
v___x_589_ = lean_box(0);
return v___x_589_;
}
else
{
lean_object* v___x_590_; 
v___x_590_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_590_, 0, v_ns_586_);
return v___x_590_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__0___boxed(lean_object* v___x_591_, lean_object* v_dropUntil_592_, lean_object* v_x_593_){
_start:
{
lean_object* v_res_594_; 
v_res_594_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__0(v___x_591_, v_dropUntil_592_, v_x_593_);
lean_dec(v_dropUntil_592_);
return v_res_594_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__1(lean_object* v_toApplicative_595_, lean_object* v_a_596_){
_start:
{
lean_object* v_toPure_597_; lean_object* v___x_598_; 
v_toPure_597_ = lean_ctor_get(v_toApplicative_595_, 1);
lean_inc(v_toPure_597_);
lean_dec_ref(v_toApplicative_595_);
v___x_598_ = lean_apply_2(v_toPure_597_, lean_box(0), v_a_596_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__3(lean_object* v_it_599_, lean_object* v_acc_600_, lean_object* v_recur_601_){
_start:
{
lean_object* v_array_602_; lean_object* v_start_603_; lean_object* v_stop_604_; lean_object* v___x_606_; uint8_t v_isShared_607_; uint8_t v_isSharedCheck_617_; 
v_array_602_ = lean_ctor_get(v_it_599_, 0);
v_start_603_ = lean_ctor_get(v_it_599_, 1);
v_stop_604_ = lean_ctor_get(v_it_599_, 2);
v_isSharedCheck_617_ = !lean_is_exclusive(v_it_599_);
if (v_isSharedCheck_617_ == 0)
{
v___x_606_ = v_it_599_;
v_isShared_607_ = v_isSharedCheck_617_;
goto v_resetjp_605_;
}
else
{
lean_inc(v_stop_604_);
lean_inc(v_start_603_);
lean_inc(v_array_602_);
lean_dec(v_it_599_);
v___x_606_ = lean_box(0);
v_isShared_607_ = v_isSharedCheck_617_;
goto v_resetjp_605_;
}
v_resetjp_605_:
{
uint8_t v___x_608_; 
v___x_608_ = lean_nat_dec_lt(v_start_603_, v_stop_604_);
if (v___x_608_ == 0)
{
lean_del_object(v___x_606_);
lean_dec(v_stop_604_);
lean_dec(v_start_603_);
lean_dec_ref(v_array_602_);
lean_dec_ref(v_recur_601_);
return v_acc_600_;
}
else
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_612_; 
v___x_609_ = lean_unsigned_to_nat(1u);
v___x_610_ = lean_nat_add(v_start_603_, v___x_609_);
lean_inc_ref(v_array_602_);
if (v_isShared_607_ == 0)
{
lean_ctor_set(v___x_606_, 1, v___x_610_);
v___x_612_ = v___x_606_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_616_; 
v_reuseFailAlloc_616_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_616_, 0, v_array_602_);
lean_ctor_set(v_reuseFailAlloc_616_, 1, v___x_610_);
lean_ctor_set(v_reuseFailAlloc_616_, 2, v_stop_604_);
v___x_612_ = v_reuseFailAlloc_616_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; 
v___x_613_ = lean_array_fget(v_array_602_, v_start_603_);
lean_dec(v_start_603_);
lean_dec_ref(v_array_602_);
v___x_614_ = lean_array_push(v_acc_600_, v___x_613_);
v___x_615_ = lean_apply_3(v_recur_601_, v___x_612_, v___x_614_, lean_box(0));
return v___x_615_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__2(lean_object* v_it_618_, lean_object* v_acc_619_, lean_object* v_hP_620_, lean_object* v_recur_621_){
_start:
{
lean_object* v_array_622_; lean_object* v_start_623_; lean_object* v_stop_624_; lean_object* v___x_626_; uint8_t v_isShared_627_; uint8_t v_isSharedCheck_637_; 
v_array_622_ = lean_ctor_get(v_it_618_, 0);
v_start_623_ = lean_ctor_get(v_it_618_, 1);
v_stop_624_ = lean_ctor_get(v_it_618_, 2);
v_isSharedCheck_637_ = !lean_is_exclusive(v_it_618_);
if (v_isSharedCheck_637_ == 0)
{
v___x_626_ = v_it_618_;
v_isShared_627_ = v_isSharedCheck_637_;
goto v_resetjp_625_;
}
else
{
lean_inc(v_stop_624_);
lean_inc(v_start_623_);
lean_inc(v_array_622_);
lean_dec(v_it_618_);
v___x_626_ = lean_box(0);
v_isShared_627_ = v_isSharedCheck_637_;
goto v_resetjp_625_;
}
v_resetjp_625_:
{
uint8_t v___x_628_; 
v___x_628_ = lean_nat_dec_lt(v_start_623_, v_stop_624_);
if (v___x_628_ == 0)
{
lean_del_object(v___x_626_);
lean_dec(v_stop_624_);
lean_dec(v_start_623_);
lean_dec_ref(v_array_622_);
lean_dec_ref(v_recur_621_);
return v_acc_619_;
}
else
{
lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_632_; 
v___x_629_ = lean_unsigned_to_nat(1u);
v___x_630_ = lean_nat_add(v_start_623_, v___x_629_);
lean_inc_ref(v_array_622_);
if (v_isShared_627_ == 0)
{
lean_ctor_set(v___x_626_, 1, v___x_630_);
v___x_632_ = v___x_626_;
goto v_reusejp_631_;
}
else
{
lean_object* v_reuseFailAlloc_636_; 
v_reuseFailAlloc_636_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_636_, 0, v_array_622_);
lean_ctor_set(v_reuseFailAlloc_636_, 1, v___x_630_);
lean_ctor_set(v_reuseFailAlloc_636_, 2, v_stop_624_);
v___x_632_ = v_reuseFailAlloc_636_;
goto v_reusejp_631_;
}
v_reusejp_631_:
{
lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; 
v___x_633_ = lean_array_fget(v_array_622_, v_start_623_);
lean_dec(v_start_623_);
lean_dec_ref(v_array_622_);
v___x_634_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_addIdents(v_acc_619_, v___x_633_);
v___x_635_ = lean_apply_4(v_recur_621_, v___x_632_, v___x_634_, lean_box(0), lean_box(0));
return v___x_635_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__4(lean_object* v___x_638_, lean_object* v___x_639_, lean_object* v_usedNames_640_, lean_object* v_toApplicative_641_, lean_object* v___x_642_, lean_object* v_a_643_, lean_object* v_x_644_, lean_object* v___y_645_){
_start:
{
lean_object* v___x_646_; uint8_t v___x_647_; 
v___x_646_ = l_Lean_TSyntax_getId(v_a_643_);
v___x_647_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v___x_638_, v___x_639_, v_usedNames_640_, v___x_646_);
if (v___x_647_ == 0)
{
lean_object* v_toPure_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
v_toPure_648_ = lean_ctor_get(v_toApplicative_641_, 1);
lean_inc(v_toPure_648_);
lean_dec_ref(v_toApplicative_641_);
v___x_649_ = lean_nat_add(v___y_645_, v___x_642_);
lean_dec(v___y_645_);
v___x_650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_650_, 0, v___x_649_);
v___x_651_ = lean_apply_2(v_toPure_648_, lean_box(0), v___x_650_);
return v___x_651_;
}
else
{
lean_object* v_toPure_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v_toPure_652_ = lean_ctor_get(v_toApplicative_641_, 1);
lean_inc(v_toPure_652_);
lean_dec_ref(v_toApplicative_641_);
v___x_653_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_653_, 0, v___y_645_);
v___x_654_ = lean_apply_2(v_toPure_652_, lean_box(0), v___x_653_);
return v___x_654_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__4___boxed(lean_object* v___x_655_, lean_object* v___x_656_, lean_object* v_usedNames_657_, lean_object* v_toApplicative_658_, lean_object* v___x_659_, lean_object* v_a_660_, lean_object* v_x_661_, lean_object* v___y_662_){
_start:
{
lean_object* v_res_663_; 
v_res_663_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__4(v___x_655_, v___x_656_, v_usedNames_657_, v_toApplicative_658_, v___x_659_, v_a_660_, v_x_661_, v___y_662_);
lean_dec(v_a_660_);
lean_dec(v___x_659_);
lean_dec_ref(v_usedNames_657_);
return v_res_663_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__5(lean_object* v___x_664_, lean_object* v___x_665_, lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_toBind_668_, lean_object* v___f_669_, lean_object* v_tac_670_){
_start:
{
lean_object* v_result_671_; lean_object* v_result_672_; lean_object* v___x_673_; lean_object* v_result_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v_result_671_ = lean_mk_empty_array_with_capacity(v___x_664_);
v_result_672_ = lean_array_push(v_result_671_, v_tac_670_);
v___x_673_ = l_Subarray_copy___redArg(v___x_665_);
v_result_674_ = l_Array_append___redArg(v_result_672_, v___x_673_);
lean_dec_ref(v___x_673_);
v___x_675_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg(v_inst_666_, v_inst_667_, v_result_674_);
v___x_676_ = lean_apply_4(v_toBind_668_, lean_box(0), lean_box(0), v___x_675_, v___f_669_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__5___boxed(lean_object* v___x_677_, lean_object* v___x_678_, lean_object* v_inst_679_, lean_object* v_inst_680_, lean_object* v_toBind_681_, lean_object* v___f_682_, lean_object* v_tac_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__5(v___x_677_, v___x_678_, v_inst_679_, v_inst_680_, v_toBind_681_, v___f_682_, v_tac_683_);
lean_dec(v___x_677_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__6(lean_object* v___x_685_, lean_object* v_info_686_, lean_object* v_x_687_){
_start:
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v___x_688_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__1___closed__0));
v___x_689_ = l_Lean_Name_mkStr2(v___x_685_, v___x_688_);
v___x_690_ = l_Lean_Syntax_node1(v_info_686_, v___x_689_, v_x_687_);
return v___x_690_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7(lean_object* v_info_692_, lean_object* v_ns_693_, lean_object* v___f_694_, size_t v___x_695_, lean_object* v___x_696_, lean_object* v_toPure_697_, lean_object* v_quotCtx_698_){
_start:
{
lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; size_t v_sz_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; 
v___x_699_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7___closed__0));
lean_inc_n(v_info_692_, 2);
v___x_700_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_700_, 0, v_info_692_);
lean_ctor_set(v___x_700_, 1, v___x_699_);
v___x_701_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__3));
v___x_702_ = lean_obj_once(&lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4, &lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4_once, _init_lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__4);
v___x_703_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__4___closed__14));
v_sz_704_ = lean_array_size(v_ns_693_);
v___x_705_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_703_, v___f_694_, v_sz_704_, v___x_695_, v_ns_693_);
v___x_706_ = l_Array_append___redArg(v___x_702_, v___x_705_);
lean_dec(v___x_705_);
v___x_707_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_707_, 0, v_info_692_);
lean_ctor_set(v___x_707_, 1, v___x_701_);
lean_ctor_set(v___x_707_, 2, v___x_706_);
v___x_708_ = l_Lean_Syntax_node2(v_info_692_, v___x_696_, v___x_700_, v___x_707_);
v___x_709_ = lean_apply_2(v_toPure_697_, lean_box(0), v___x_708_);
return v___x_709_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7___boxed(lean_object* v_info_710_, lean_object* v_ns_711_, lean_object* v___f_712_, lean_object* v___x_713_, lean_object* v___x_714_, lean_object* v_toPure_715_, lean_object* v_quotCtx_716_){
_start:
{
size_t v___x_4163__boxed_717_; lean_object* v_res_718_; 
v___x_4163__boxed_717_ = lean_unbox_usize(v___x_713_);
lean_dec(v___x_713_);
v_res_718_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7(v_info_710_, v_ns_711_, v___f_712_, v___x_4163__boxed_717_, v___x_714_, v_toPure_715_, v_quotCtx_716_);
lean_dec(v_quotCtx_716_);
return v_res_718_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__9(lean_object* v___x_719_, lean_object* v_ns_720_, size_t v___x_721_, lean_object* v___x_722_, lean_object* v_toPure_723_, lean_object* v_toBind_724_, lean_object* v_getContext_725_, lean_object* v_getCurrMacroScope_726_, lean_object* v_info_727_){
_start:
{
lean_object* v___f_728_; lean_object* v___x_729_; lean_object* v___f_730_; lean_object* v___f_731_; lean_object* v___x_732_; 
lean_inc(v_info_727_);
v___f_728_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__6), 3, 2);
lean_closure_set(v___f_728_, 0, v___x_719_);
lean_closure_set(v___f_728_, 1, v_info_727_);
v___x_729_ = lean_box_usize(v___x_721_);
v___f_730_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__7___boxed), 7, 6);
lean_closure_set(v___f_730_, 0, v_info_727_);
lean_closure_set(v___f_730_, 1, v_ns_720_);
lean_closure_set(v___f_730_, 2, v___f_728_);
lean_closure_set(v___f_730_, 3, v___x_729_);
lean_closure_set(v___f_730_, 4, v___x_722_);
lean_closure_set(v___f_730_, 5, v_toPure_723_);
lean_inc(v_toBind_724_);
v___f_731_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_731_, 0, v_toBind_724_);
lean_closure_set(v___f_731_, 1, v_getContext_725_);
lean_closure_set(v___f_731_, 2, v___f_730_);
v___x_732_ = lean_apply_4(v_toBind_724_, lean_box(0), lean_box(0), v_getCurrMacroScope_726_, v___f_731_);
return v___x_732_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__9___boxed(lean_object* v___x_733_, lean_object* v_ns_734_, lean_object* v___x_735_, lean_object* v___x_736_, lean_object* v_toPure_737_, lean_object* v_toBind_738_, lean_object* v_getContext_739_, lean_object* v_getCurrMacroScope_740_, lean_object* v_info_741_){
_start:
{
size_t v___x_4198__boxed_742_; lean_object* v_res_743_; 
v___x_4198__boxed_742_ = lean_unbox_usize(v___x_735_);
lean_dec(v___x_735_);
v_res_743_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__9(v___x_733_, v_ns_734_, v___x_4198__boxed_742_, v___x_736_, v_toPure_737_, v_toBind_738_, v_getContext_739_, v_getCurrMacroScope_740_, v_info_741_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__8(uint8_t v___x_744_, lean_object* v_toPure_745_, lean_object* v_____do__lift_746_){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; 
v___x_747_ = l_Lean_SourceInfo_fromRef(v_____do__lift_746_, v___x_744_);
v___x_748_ = lean_apply_2(v_toPure_745_, lean_box(0), v___x_747_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__8___boxed(lean_object* v___x_749_, lean_object* v_toPure_750_, lean_object* v_____do__lift_751_){
_start:
{
uint8_t v___x_4219__boxed_752_; lean_object* v_res_753_; 
v___x_4219__boxed_752_ = lean_unbox(v___x_749_);
v_res_753_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__8(v___x_4219__boxed_752_, v_toPure_750_, v_____do__lift_751_);
lean_dec(v_____do__lift_751_);
return v_res_753_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__10(lean_object* v_dropUntil_754_, lean_object* v_val_755_, lean_object* v_inst_756_, lean_object* v_toApplicative_757_, lean_object* v___f_758_, lean_object* v___x_759_, size_t v___x_760_, lean_object* v___x_761_, lean_object* v_toBind_762_, lean_object* v___f_763_, lean_object* v___x_764_, lean_object* v_inst_765_, lean_object* v___f_766_, lean_object* v_x_767_, lean_object* v_____s_768_){
_start:
{
uint8_t v___x_769_; 
v___x_769_ = lean_nat_dec_eq(v_____s_768_, v_dropUntil_754_);
if (v___x_769_ == 0)
{
lean_object* v___x_770_; uint8_t v___x_771_; 
lean_dec(v_x_767_);
v___x_770_ = lean_array_get_size(v_val_755_);
v___x_771_ = lean_nat_dec_eq(v_____s_768_, v___x_770_);
if (v___x_771_ == 0)
{
lean_object* v_toMonadRef_772_; lean_object* v_getCurrMacroScope_773_; lean_object* v_getContext_774_; lean_object* v_getRef_775_; lean_object* v_toPure_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v_ns_779_; lean_object* v___x_780_; lean_object* v___f_781_; lean_object* v___x_782_; lean_object* v___f_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; 
lean_dec(v___f_766_);
lean_dec_ref(v_inst_765_);
lean_dec_ref(v___x_764_);
v_toMonadRef_772_ = lean_ctor_get(v_inst_756_, 0);
lean_inc_ref(v_toMonadRef_772_);
v_getCurrMacroScope_773_ = lean_ctor_get(v_inst_756_, 1);
lean_inc(v_getCurrMacroScope_773_);
v_getContext_774_ = lean_ctor_get(v_inst_756_, 2);
lean_inc(v_getContext_774_);
lean_dec_ref(v_inst_756_);
v_getRef_775_ = lean_ctor_get(v_toMonadRef_772_, 0);
lean_inc(v_getRef_775_);
lean_dec_ref(v_toMonadRef_772_);
v_toPure_776_ = lean_ctor_get(v_toApplicative_757_, 1);
lean_inc_n(v_toPure_776_, 2);
lean_dec_ref(v_toApplicative_757_);
v___x_777_ = l_Array_toSubarray___redArg(v_val_755_, v_____s_768_, v___x_770_);
v___x_778_ = lean_mk_empty_array_with_capacity(v_dropUntil_754_);
v_ns_779_ = l___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___redArg(v___f_758_, v___x_777_, v___x_778_);
v___x_780_ = lean_box_usize(v___x_760_);
lean_inc_n(v_toBind_762_, 3);
v___f_781_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__9___boxed), 9, 8);
lean_closure_set(v___f_781_, 0, v___x_759_);
lean_closure_set(v___f_781_, 1, v_ns_779_);
lean_closure_set(v___f_781_, 2, v___x_780_);
lean_closure_set(v___f_781_, 3, v___x_761_);
lean_closure_set(v___f_781_, 4, v_toPure_776_);
lean_closure_set(v___f_781_, 5, v_toBind_762_);
lean_closure_set(v___f_781_, 6, v_getContext_774_);
lean_closure_set(v___f_781_, 7, v_getCurrMacroScope_773_);
v___x_782_ = lean_box(v___x_771_);
v___f_783_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__8___boxed), 3, 2);
lean_closure_set(v___f_783_, 0, v___x_782_);
lean_closure_set(v___f_783_, 1, v_toPure_776_);
v___x_784_ = lean_apply_4(v_toBind_762_, lean_box(0), lean_box(0), v_getRef_775_, v___f_783_);
v___x_785_ = lean_apply_4(v_toBind_762_, lean_box(0), lean_box(0), v___x_784_, v___f_781_);
v___x_786_ = lean_apply_4(v_toBind_762_, lean_box(0), lean_box(0), v___x_785_, v___f_763_);
return v___x_786_;
}
else
{
lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; 
lean_dec(v_____s_768_);
lean_dec(v___f_763_);
lean_dec(v___x_761_);
lean_dec_ref(v___x_759_);
lean_dec_ref(v___f_758_);
lean_dec_ref(v_toApplicative_757_);
lean_dec_ref(v_val_755_);
v___x_787_ = l_Subarray_copy___redArg(v___x_764_);
v___x_788_ = lp_aesop___private_Aesop_Script_OptimizeSyntax_0__Aesop_optimizeInitialRenameI_tacsToTacticSeq___redArg(v_inst_765_, v_inst_756_, v___x_787_);
v___x_789_ = lean_apply_4(v_toBind_762_, lean_box(0), lean_box(0), v___x_788_, v___f_766_);
return v___x_789_;
}
}
else
{
lean_object* v_toPure_790_; lean_object* v___x_791_; 
lean_dec(v_____s_768_);
lean_dec(v___f_766_);
lean_dec_ref(v_inst_765_);
lean_dec_ref(v___x_764_);
lean_dec(v___f_763_);
lean_dec(v_toBind_762_);
lean_dec(v___x_761_);
lean_dec_ref(v___x_759_);
lean_dec_ref(v___f_758_);
lean_dec_ref(v_inst_756_);
lean_dec_ref(v_val_755_);
v_toPure_790_ = lean_ctor_get(v_toApplicative_757_, 1);
lean_inc(v_toPure_790_);
lean_dec_ref(v_toApplicative_757_);
v___x_791_ = lean_apply_2(v_toPure_790_, lean_box(0), v_x_767_);
return v___x_791_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__10___boxed(lean_object* v_dropUntil_792_, lean_object* v_val_793_, lean_object* v_inst_794_, lean_object* v_toApplicative_795_, lean_object* v___f_796_, lean_object* v___x_797_, lean_object* v___x_798_, lean_object* v___x_799_, lean_object* v_toBind_800_, lean_object* v___f_801_, lean_object* v___x_802_, lean_object* v_inst_803_, lean_object* v___f_804_, lean_object* v_x_805_, lean_object* v_____s_806_){
_start:
{
size_t v___x_4233__boxed_807_; lean_object* v_res_808_; 
v___x_4233__boxed_807_ = lean_unbox_usize(v___x_798_);
lean_dec(v___x_798_);
v_res_808_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__10(v_dropUntil_792_, v_val_793_, v_inst_794_, v_toApplicative_795_, v___f_796_, v___x_797_, v___x_4233__boxed_807_, v___x_799_, v_toBind_800_, v___f_801_, v___x_802_, v_inst_803_, v___f_804_, v_x_805_, v_____s_806_);
lean_dec(v_dropUntil_792_);
return v_res_808_;
}
}
static lean_object* _init_lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__5(void){
_start:
{
lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; 
v___x_816_ = lean_box(0);
v___x_817_ = lean_unsigned_to_nat(16u);
v___x_818_ = lean_mk_array(v___x_817_, v___x_816_);
return v___x_818_;
}
}
static lean_object* _init_lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__6(void){
_start:
{
lean_object* v___x_819_; lean_object* v_dropUntil_820_; lean_object* v___x_821_; 
v___x_819_ = lean_obj_once(&lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__5, &lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__5_once, _init_lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__5);
v_dropUntil_820_ = lean_unsigned_to_nat(0u);
v___x_821_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_821_, 0, v_dropUntil_820_);
lean_ctor_set(v___x_821_, 1, v___x_819_);
return v___x_821_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI___redArg(lean_object* v_inst_822_, lean_object* v_inst_823_, lean_object* v_x_824_){
_start:
{
lean_object* v___x_825_; lean_object* v___x_826_; uint8_t v___x_827_; 
v___x_825_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__0));
v___x_826_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__8));
lean_inc(v_x_824_);
v___x_827_ = l_Lean_Syntax_isOfKind(v_x_824_, v___x_826_);
if (v___x_827_ == 0)
{
lean_object* v_toApplicative_828_; lean_object* v_toPure_829_; lean_object* v___x_830_; 
lean_dec_ref(v_inst_823_);
v_toApplicative_828_ = lean_ctor_get(v_inst_822_, 0);
lean_inc_ref(v_toApplicative_828_);
lean_dec_ref(v_inst_822_);
v_toPure_829_ = lean_ctor_get(v_toApplicative_828_, 1);
lean_inc(v_toPure_829_);
lean_dec_ref(v_toApplicative_828_);
v___x_830_ = lean_apply_2(v_toPure_829_, lean_box(0), v_x_824_);
return v___x_830_;
}
else
{
lean_object* v_dropUntil_831_; lean_object* v___x_832_; lean_object* v___x_833_; uint8_t v___x_834_; 
v_dropUntil_831_ = lean_unsigned_to_nat(0u);
v___x_832_ = l_Lean_Syntax_getArg(v_x_824_, v_dropUntil_831_);
v___x_833_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__10));
lean_inc(v___x_832_);
v___x_834_ = l_Lean_Syntax_isOfKind(v___x_832_, v___x_833_);
if (v___x_834_ == 0)
{
lean_object* v_toApplicative_835_; lean_object* v_toPure_836_; lean_object* v___x_837_; 
lean_dec(v___x_832_);
lean_dec_ref(v_inst_823_);
v_toApplicative_835_ = lean_ctor_get(v_inst_822_, 0);
lean_inc_ref(v_toApplicative_835_);
lean_dec_ref(v_inst_822_);
v_toPure_836_ = lean_ctor_get(v_toApplicative_835_, 1);
lean_inc(v_toPure_836_);
lean_dec_ref(v_toApplicative_835_);
v___x_837_ = lean_apply_2(v_toPure_836_, lean_box(0), v_x_824_);
return v___x_837_;
}
else
{
lean_object* v___x_838_; lean_object* v_tacs_839_; lean_object* v_a_840_; lean_object* v___x_841_; uint8_t v___x_842_; 
v___x_838_ = l_Lean_Syntax_getArg(v___x_832_, v_dropUntil_831_);
lean_dec(v___x_832_);
v_tacs_839_ = l_Lean_Syntax_getArgs(v___x_838_);
lean_dec(v___x_838_);
v_a_840_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_tacs_839_);
lean_dec_ref(v_tacs_839_);
v___x_841_ = lean_array_get_size(v_a_840_);
v___x_842_ = lean_nat_dec_lt(v_dropUntil_831_, v___x_841_);
if (v___x_842_ == 0)
{
lean_object* v_toApplicative_843_; lean_object* v_toPure_844_; lean_object* v___x_845_; 
lean_dec_ref(v_a_840_);
lean_dec_ref(v_inst_823_);
v_toApplicative_843_ = lean_ctor_get(v_inst_822_, 0);
lean_inc_ref(v_toApplicative_843_);
lean_dec_ref(v_inst_822_);
v_toPure_844_ = lean_ctor_get(v_toApplicative_843_, 1);
lean_inc(v_toPure_844_);
lean_dec_ref(v_toApplicative_843_);
v___x_845_ = lean_apply_2(v_toPure_844_, lean_box(0), v_x_824_);
return v___x_845_;
}
else
{
lean_object* v___x_846_; lean_object* v___x_847_; uint8_t v___x_848_; 
v___x_846_ = lean_array_fget(v_a_840_, v_dropUntil_831_);
v___x_847_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__14));
lean_inc(v___x_846_);
v___x_848_ = l_Lean_Syntax_isOfKind(v___x_846_, v___x_847_);
if (v___x_848_ == 0)
{
lean_object* v_toApplicative_849_; lean_object* v_toPure_850_; lean_object* v___x_851_; 
lean_dec(v___x_846_);
lean_dec_ref(v_a_840_);
lean_dec_ref(v_inst_823_);
v_toApplicative_849_ = lean_ctor_get(v_inst_822_, 0);
lean_inc_ref(v_toApplicative_849_);
lean_dec_ref(v_inst_822_);
v_toPure_850_ = lean_ctor_get(v_toApplicative_849_, 1);
lean_inc(v_toPure_850_);
lean_dec_ref(v_toApplicative_849_);
v___x_851_ = lean_apply_2(v_toPure_850_, lean_box(0), v_x_824_);
return v___x_851_;
}
else
{
lean_object* v___f_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; size_t v_sz_857_; size_t v___x_858_; lean_object* v___x_859_; 
v___f_852_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__0));
v___x_853_ = lean_unsigned_to_nat(1u);
v___x_854_ = l_Lean_Syntax_getArg(v___x_846_, v___x_853_);
lean_dec(v___x_846_);
v___x_855_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___closed__25));
v___x_856_ = l_Lean_Syntax_getArgs(v___x_854_);
lean_dec(v___x_854_);
v_sz_857_ = lean_array_size(v___x_856_);
v___x_858_ = ((size_t)0ULL);
v___x_859_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_855_, v___f_852_, v_sz_857_, v___x_858_, v___x_856_);
if (lean_obj_tag(v___x_859_) == 0)
{
lean_object* v_toApplicative_860_; lean_object* v_toPure_861_; lean_object* v___x_862_; 
lean_dec_ref(v_a_840_);
lean_dec_ref(v_inst_823_);
v_toApplicative_860_ = lean_ctor_get(v_inst_822_, 0);
lean_inc_ref(v_toApplicative_860_);
lean_dec_ref(v_inst_822_);
v_toPure_861_ = lean_ctor_get(v_toApplicative_860_, 1);
lean_inc(v_toPure_861_);
lean_dec_ref(v_toApplicative_860_);
v___x_862_ = lean_apply_2(v_toPure_861_, lean_box(0), v_x_824_);
return v___x_862_;
}
else
{
lean_object* v_val_863_; lean_object* v_toApplicative_864_; lean_object* v_toBind_865_; lean_object* v___f_866_; lean_object* v___f_867_; lean_object* v___f_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v_usedNames_873_; lean_object* v___f_874_; lean_object* v___f_875_; lean_object* v___x_876_; lean_object* v___f_877_; size_t v_sz_878_; lean_object* v___x_879_; lean_object* v___x_880_; 
v_val_863_ = lean_ctor_get(v___x_859_, 0);
lean_inc_n(v_val_863_, 2);
lean_dec_ref_known(v___x_859_, 1);
v_toApplicative_864_ = lean_ctor_get(v_inst_822_, 0);
v_toBind_865_ = lean_ctor_get(v_inst_822_, 1);
lean_inc_n(v_toBind_865_, 3);
lean_inc_ref_n(v_toApplicative_864_, 3);
v___f_866_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__1), 2, 1);
lean_closure_set(v___f_866_, 0, v_toApplicative_864_);
v___f_867_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__1));
v___f_868_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__2));
v___x_869_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__3));
v___x_870_ = ((lean_object*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__4));
v___x_871_ = lean_obj_once(&lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__6, &lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__6_once, _init_lp_aesop_Aesop_optimizeInitialRenameI___redArg___closed__6);
v___x_872_ = l_Array_toSubarray___redArg(v_a_840_, v___x_853_, v___x_841_);
lean_inc_ref_n(v___x_872_, 2);
v_usedNames_873_ = l_WellFounded_opaqueFix_u2083___redArg(v___f_868_, v___x_872_, v___x_871_, lean_box(0));
v___f_874_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__4___boxed), 8, 5);
lean_closure_set(v___f_874_, 0, v___x_869_);
lean_closure_set(v___f_874_, 1, v___x_870_);
lean_closure_set(v___f_874_, 2, v_usedNames_873_);
lean_closure_set(v___f_874_, 3, v_toApplicative_864_);
lean_closure_set(v___f_874_, 4, v___x_853_);
lean_inc_ref(v___f_866_);
lean_inc_ref(v_inst_823_);
lean_inc_ref_n(v_inst_822_, 2);
v___f_875_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__5___boxed), 7, 6);
lean_closure_set(v___f_875_, 0, v___x_841_);
lean_closure_set(v___f_875_, 1, v___x_872_);
lean_closure_set(v___f_875_, 2, v_inst_822_);
lean_closure_set(v___f_875_, 3, v_inst_823_);
lean_closure_set(v___f_875_, 4, v_toBind_865_);
lean_closure_set(v___f_875_, 5, v___f_866_);
v___x_876_ = ((lean_object*)(lp_aesop_Aesop_optimizeFocusRenameI___redArg___lam__12___boxed__const__1));
v___f_877_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeInitialRenameI___redArg___lam__10___boxed), 15, 14);
lean_closure_set(v___f_877_, 0, v_dropUntil_831_);
lean_closure_set(v___f_877_, 1, v_val_863_);
lean_closure_set(v___f_877_, 2, v_inst_823_);
lean_closure_set(v___f_877_, 3, v_toApplicative_864_);
lean_closure_set(v___f_877_, 4, v___f_867_);
lean_closure_set(v___f_877_, 5, v___x_825_);
lean_closure_set(v___f_877_, 6, v___x_876_);
lean_closure_set(v___f_877_, 7, v___x_847_);
lean_closure_set(v___f_877_, 8, v_toBind_865_);
lean_closure_set(v___f_877_, 9, v___f_875_);
lean_closure_set(v___f_877_, 10, v___x_872_);
lean_closure_set(v___f_877_, 11, v_inst_822_);
lean_closure_set(v___f_877_, 12, v___f_866_);
lean_closure_set(v___f_877_, 13, v_x_824_);
v_sz_878_ = lean_array_size(v_val_863_);
v___x_879_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_822_, v_val_863_, v___f_874_, v_sz_878_, v___x_858_, v_dropUntil_831_);
v___x_880_ = lean_apply_4(v_toBind_865_, lean_box(0), lean_box(0), v___x_879_, v___f_877_);
return v___x_880_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeInitialRenameI(lean_object* v_m_881_, lean_object* v_inst_882_, lean_object* v_inst_883_, lean_object* v_x_884_){
_start:
{
lean_object* v___x_885_; 
v___x_885_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg(v_inst_882_, v_inst_883_, v_x_884_);
return v___x_885_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___redArg___lam__0(lean_object* v_toPure_886_, lean_object* v_stx_887_){
_start:
{
lean_object* v___x_888_; 
v___x_888_ = lean_apply_2(v_toPure_886_, lean_box(0), v_stx_887_);
return v___x_888_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___redArg___lam__1(lean_object* v_inst_889_, lean_object* v_inst_890_, lean_object* v_toBind_891_, lean_object* v___f_892_, lean_object* v_stx_893_){
_start:
{
lean_object* v___x_894_; lean_object* v___x_895_; 
v___x_894_ = lp_aesop_Aesop_optimizeInitialRenameI___redArg(v_inst_889_, v_inst_890_, v_stx_893_);
v___x_895_ = lean_apply_4(v_toBind_891_, lean_box(0), lean_box(0), v___x_894_, v___f_892_);
return v___x_895_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___redArg(lean_object* v_inst_896_, lean_object* v_inst_897_, lean_object* v_stx_898_){
_start:
{
lean_object* v_toApplicative_899_; lean_object* v_toBind_900_; lean_object* v_toPure_901_; lean_object* v___x_902_; lean_object* v___f_903_; lean_object* v___f_904_; lean_object* v___x_905_; 
v_toApplicative_899_ = lean_ctor_get(v_inst_896_, 0);
v_toBind_900_ = lean_ctor_get(v_inst_896_, 1);
lean_inc_n(v_toBind_900_, 2);
v_toPure_901_ = lean_ctor_get(v_toApplicative_899_, 1);
lean_inc_ref(v_inst_897_);
lean_inc_ref(v_inst_896_);
v___x_902_ = lp_aesop_Aesop_optimizeFocusRenameI___redArg(v_inst_896_, v_inst_897_, v_stx_898_);
lean_inc(v_toPure_901_);
v___f_903_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeSyntax___redArg___lam__0), 2, 1);
lean_closure_set(v___f_903_, 0, v_toPure_901_);
v___f_904_ = lean_alloc_closure((void*)(lp_aesop_Aesop_optimizeSyntax___redArg___lam__1), 5, 4);
lean_closure_set(v___f_904_, 0, v_inst_896_);
lean_closure_set(v___f_904_, 1, v_inst_897_);
lean_closure_set(v___f_904_, 2, v_toBind_900_);
lean_closure_set(v___f_904_, 3, v___f_903_);
v___x_905_ = lean_apply_4(v_toBind_900_, lean_box(0), lean_box(0), v___x_902_, v___f_904_);
return v___x_905_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax(lean_object* v_m_906_, lean_object* v_inst_907_, lean_object* v_inst_908_, lean_object* v_kind_909_, lean_object* v_stx_910_){
_start:
{
lean_object* v___x_911_; 
v___x_911_ = lp_aesop_Aesop_optimizeSyntax___redArg(v_inst_907_, v_inst_908_, v_stx_910_);
return v___x_911_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_optimizeSyntax___boxed(lean_object* v_m_912_, lean_object* v_inst_913_, lean_object* v_inst_914_, lean_object* v_kind_915_, lean_object* v_stx_916_){
_start:
{
lean_object* v_res_917_; 
v_res_917_ = lp_aesop_Aesop_optimizeSyntax(v_m_912_, v_inst_913_, v_inst_914_, v_kind_915_, v_stx_916_);
lean_dec(v_kind_915_);
return v_res_917_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_HashSet_Basic(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_OptimizeSyntax(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_HashSet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Term_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_OptimizeSyntax(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Term_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Std_Data_HashSet_Basic(uint8_t builtin);
lean_object* initialize_Lean_Parser_Term_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_OptimizeSyntax(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_HashSet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Term_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_OptimizeSyntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_OptimizeSyntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_OptimizeSyntax(builtin);
}
#ifdef __cplusplus
}
#endif
