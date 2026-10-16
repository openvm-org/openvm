// Lean compiler output
// Module: Aesop.Script.CtorNames
// Imports: public import Init public meta import Init public import Lean.Meta.Tactic.Induction public import Aesop.Util.Basic
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_array_mk(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Environment_findAsync_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_AsyncConstantInfo_toConstantInfo(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_binderInfo(lean_object*);
uint8_t l_Lean_BinderInfo_isExplicit(uint8_t);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
extern lean_object* l_Lean_firstFrontendMacroScope;
lean_object* lp_aesop_Aesop_getUnusedNames(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "rcasesPatLo"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__3_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4_value_aux_2),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(133, 222, 245, 138, 122, 92, 170, 214)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "rcasesPatMed"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__5_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6_value_aux_2),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(253, 13, 65, 195, 228, 27, 47, 149)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__7_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rcasesPat"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__9_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "one"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__10_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value_aux_2),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(162, 181, 165, 225, 136, 177, 169, 19)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value_aux_3),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(186, 152, 172, 228, 11, 240, 156, 168)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toRCasesPat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__0;
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toRCasesPat___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__1;
static const lean_string_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tuple"};
static const lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__2 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value_aux_2),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(162, 181, 165, 225, 136, 177, 169, 19)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value_aux_3),((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__2_value),LEAN_SCALAR_PTR_LITERAL(50, 241, 13, 230, 132, 227, 26, 91)}};
static const lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__4 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__4_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__5 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__5_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__6 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7;
static const lean_string_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__8 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value_aux_2),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(162, 181, 165, 225, 136, 177, 169, 19)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value_aux_3),((lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__8_value),LEAN_SCALAR_PTR_LITERAL(176, 12, 240, 143, 52, 56, 179, 56)}};
static const lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toRCasesPat___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__10 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11;
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toRCasesPat___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__12;
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toRCasesPat___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat___closed__13;
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_CtorNames_0__Aesop_CtorNames_nameBase(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toInductionAltLHS_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toInductionAltLHS_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "inductionAltLHS"};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__0 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(58, 206, 3, 35, 121, 94, 13, 140)}};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__2 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__2_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__3 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__3_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__4 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__5;
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS(lean_object*);
static const lean_string_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "inductionAlt"};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__0 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 42, 154, 153, 222, 213, 69, 136)}};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__2 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_CtorNames_toInductionAlt___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__3;
static const lean_string_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__4 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5_value_aux_2),((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__6 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7_value_aux_2),((lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7_value;
static const lean_string_object lp_aesop_Aesop_CtorNames_toInductionAlt___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___closed__8 = (const lean_object*)&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__8_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toAltVarNames(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_mkFreshArgNames(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToRCasesPats_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToRCasesPats_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ctorNamesToRCasesPats(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToInductionAlts_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToInductionAlts_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ctorNamesToInductionAlts___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inductionAlts"};
static const lean_object* lp_aesop_Aesop_ctorNamesToInductionAlts___closed__0 = (const lean_object*)&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1_value_aux_1),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 186, 227, 253, 35, 189, 199, 190)}};
static const lean_object* lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1 = (const lean_object*)&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1_value;
static const lean_string_object lp_aesop_Aesop_ctorNamesToInductionAlts___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_aesop_Aesop_ctorNamesToInductionAlts___closed__2 = (const lean_object*)&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_ctorNamesToInductionAlts___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ctorNamesToInductionAlts___closed__3;
static lean_once_cell_t lp_aesop_Aesop_ctorNamesToInductionAlts___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ctorNamesToInductionAlts___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ctorNamesToInductionAlts(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__0;
static const lean_closure_object lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__1 = (const lean_object*)&lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__1_value;
static const lean_closure_object lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__2 = (const lean_object*)&lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__2_value;
static const lean_closure_object lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__3 = (const lean_object*)&lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__3_value;
static const lean_closure_object lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__4 = (const lean_object*)&lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__0 = (const lean_object*)&lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__0_value;
static lean_once_cell_t lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__1;
static const lean_string_object lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "` is not a constructor"};
static const lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__2 = (const lean_object*)&lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__2_value;
static lean_once_cell_t lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__3;
static const lean_string_object lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Lean.MonadEnv"};
static const lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__4 = (const lean_object*)&lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__4_value;
static const lean_string_object lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Lean.isCtor\?"};
static const lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__5 = (const lean_object*)&lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__5_value;
static const lean_string_object lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__6 = (const lean_object*)&lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__6_value;
static lean_once_cell_t lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_mkCtorNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_mkCtorNames___closed__0 = (const lean_object*)&lp_aesop_Aesop_mkCtorNames___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_mkCtorNames___closed__1;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__2;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__3;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__4;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__5;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__6;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__7;
static const lean_array_object lp_aesop_Aesop_mkCtorNames___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_mkCtorNames___closed__8 = (const lean_object*)&lp_aesop_Aesop_mkCtorNames___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__9;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__10;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__11;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__12;
static lean_once_cell_t lp_aesop_Aesop_mkCtorNames___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_mkCtorNames___closed__13;
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkCtorNames(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkCtorNames___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12(void){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = l_Array_mkArray0(lean_box(0));
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0(size_t v_sz_28_, size_t v_i_29_, lean_object* v_bs_30_, lean_object* v___y_31_, lean_object* v___y_32_){
_start:
{
uint8_t v___x_33_; 
v___x_33_ = lean_usize_dec_lt(v_i_29_, v_sz_28_);
if (v___x_33_ == 0)
{
lean_object* v___x_34_; 
v___x_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_34_, 0, v_bs_30_);
lean_ctor_set(v___x_34_, 1, v___y_32_);
return v___x_34_;
}
else
{
lean_object* v_ref_35_; lean_object* v_v_36_; lean_object* v___x_37_; lean_object* v_bs_x27_38_; uint8_t v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; size_t v___x_52_; size_t v___x_53_; lean_object* v___x_54_; 
v_ref_35_ = lean_ctor_get(v___y_31_, 0);
v_v_36_ = lean_array_uget(v_bs_30_, v_i_29_);
v___x_37_ = lean_unsigned_to_nat(0u);
v_bs_x27_38_ = lean_array_uset(v_bs_30_, v_i_29_, v___x_37_);
v___x_39_ = 0;
v___x_40_ = l_Lean_SourceInfo_fromRef(v_ref_35_, v___x_39_);
v___x_41_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__4));
v___x_42_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6));
v___x_43_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_44_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__11));
v___x_45_ = l_Lean_mkIdent(v_v_36_);
lean_inc_n(v___x_40_, 4);
v___x_46_ = l_Lean_Syntax_node1(v___x_40_, v___x_44_, v___x_45_);
v___x_47_ = l_Lean_Syntax_node1(v___x_40_, v___x_43_, v___x_46_);
v___x_48_ = l_Lean_Syntax_node1(v___x_40_, v___x_42_, v___x_47_);
v___x_49_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_50_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_50_, 0, v___x_40_);
lean_ctor_set(v___x_50_, 1, v___x_43_);
lean_ctor_set(v___x_50_, 2, v___x_49_);
v___x_51_ = l_Lean_Syntax_node2(v___x_40_, v___x_41_, v___x_48_, v___x_50_);
v___x_52_ = ((size_t)1ULL);
v___x_53_ = lean_usize_add(v_i_29_, v___x_52_);
v___x_54_ = lean_array_uset(v_bs_x27_38_, v_i_29_, v___x_51_);
v_i_29_ = v___x_53_;
v_bs_30_ = v___x_54_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___boxed(lean_object* v_sz_56_, lean_object* v_i_57_, lean_object* v_bs_58_, lean_object* v___y_59_, lean_object* v___y_60_){
_start:
{
size_t v_sz_boxed_61_; size_t v_i_boxed_62_; lean_object* v_res_63_; 
v_sz_boxed_61_ = lean_unbox_usize(v_sz_56_);
lean_dec(v_sz_56_);
v_i_boxed_62_ = lean_unbox_usize(v_i_57_);
lean_dec(v_i_57_);
v_res_63_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0(v_sz_boxed_61_, v_i_boxed_62_, v_bs_58_, v___y_59_, v___y_60_);
lean_dec_ref(v___y_59_);
return v_res_63_;
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__0(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_64_ = l_Lean_firstFrontendMacroScope;
v___x_65_ = lean_box(0);
v___x_66_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v___x_64_);
return v___x_66_;
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__1(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_67_ = lean_unsigned_to_nat(1u);
v___x_68_ = l_Lean_firstFrontendMacroScope;
v___x_69_ = lean_nat_add(v___x_68_, v___x_67_);
return v___x_69_;
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7(void){
_start:
{
uint8_t v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_80_ = 0;
v___x_81_ = lean_box(0);
v___x_82_ = l_Lean_SourceInfo_fromRef(v___x_81_, v___x_80_);
return v___x_82_;
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11(void){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_91_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__10));
v___x_92_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_93_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v___x_91_);
return v___x_93_;
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__12(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_94_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__4));
v___x_95_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_96_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v___x_94_);
return v___x_96_;
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__13(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_97_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__6));
v___x_98_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_99_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v___x_97_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat(lean_object* v_cn_100_){
_start:
{
lean_object* v_args_101_; uint8_t v_hasImplicitArg_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; size_t v_sz_106_; size_t v___x_107_; lean_object* v___x_108_; 
v_args_101_ = lean_ctor_get(v_cn_100_, 1);
lean_inc_ref(v_args_101_);
v_hasImplicitArg_102_ = lean_ctor_get_uint8(v_cn_100_, sizeof(void*)*2);
lean_dec_ref(v_cn_100_);
v___x_103_ = lean_box(0);
v___x_104_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__0, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__0_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__0);
v___x_105_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__1, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__1_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__1);
v_sz_106_ = lean_array_size(v_args_101_);
v___x_107_ = ((size_t)0ULL);
v___x_108_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0(v_sz_106_, v___x_107_, v_args_101_, v___x_104_, v___x_105_);
if (v_hasImplicitArg_102_ == 0)
{
lean_object* v_fst_109_; lean_object* v___x_111_; uint8_t v_isShared_112_; uint8_t v_isSharedCheck_128_; 
v_fst_109_ = lean_ctor_get(v___x_108_, 0);
v_isSharedCheck_128_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_128_ == 0)
{
lean_object* v_unused_129_; 
v_unused_129_ = lean_ctor_get(v___x_108_, 1);
lean_dec(v_unused_129_);
v___x_111_ = v___x_108_;
v_isShared_112_ = v_isSharedCheck_128_;
goto v_resetjp_110_;
}
else
{
lean_inc(v_fst_109_);
lean_dec(v___x_108_);
v___x_111_ = lean_box(0);
v_isShared_112_ = v_isSharedCheck_128_;
goto v_resetjp_110_;
}
v_resetjp_110_:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_117_; 
v___x_113_ = l_Lean_SourceInfo_fromRef(v___x_103_, v_hasImplicitArg_102_);
v___x_114_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3));
v___x_115_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__4));
lean_inc(v___x_113_);
if (v_isShared_112_ == 0)
{
lean_ctor_set_tag(v___x_111_, 2);
lean_ctor_set(v___x_111_, 1, v___x_115_);
lean_ctor_set(v___x_111_, 0, v___x_113_);
v___x_117_ = v___x_111_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_127_; 
v_reuseFailAlloc_127_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_127_, 0, v___x_113_);
lean_ctor_set(v_reuseFailAlloc_127_, 1, v___x_115_);
v___x_117_ = v_reuseFailAlloc_127_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_118_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_119_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_120_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__5));
v___x_121_ = l_Lean_Syntax_SepArray_ofElems(v___x_120_, v_fst_109_);
lean_dec(v_fst_109_);
v___x_122_ = l_Array_append___redArg(v___x_119_, v___x_121_);
lean_dec_ref(v___x_121_);
lean_inc_n(v___x_113_, 2);
v___x_123_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_123_, 0, v___x_113_);
lean_ctor_set(v___x_123_, 1, v___x_118_);
lean_ctor_set(v___x_123_, 2, v___x_122_);
v___x_124_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__6));
v___x_125_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_113_);
lean_ctor_set(v___x_125_, 1, v___x_124_);
v___x_126_ = l_Lean_Syntax_node3(v___x_113_, v___x_114_, v___x_117_, v___x_123_, v___x_125_);
return v___x_126_;
}
}
}
else
{
lean_object* v_fst_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v_fst_130_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_fst_130_);
lean_dec_ref(v___x_108_);
v___x_131_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_132_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__9));
v___x_133_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11);
v___x_134_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__3));
v___x_135_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__12, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__12_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__12);
v___x_136_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_137_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_138_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toRCasesPat___closed__5));
v___x_139_ = l_Lean_Syntax_SepArray_ofElems(v___x_138_, v_fst_130_);
lean_dec(v_fst_130_);
v___x_140_ = l_Array_append___redArg(v___x_137_, v___x_139_);
lean_dec_ref(v___x_139_);
v___x_141_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_141_, 0, v___x_131_);
lean_ctor_set(v___x_141_, 1, v___x_136_);
lean_ctor_set(v___x_141_, 2, v___x_140_);
v___x_142_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__13, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__13_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__13);
v___x_143_ = l_Lean_Syntax_node3(v___x_131_, v___x_134_, v___x_135_, v___x_141_, v___x_142_);
v___x_144_ = l_Lean_Syntax_node2(v___x_131_, v___x_132_, v___x_133_, v___x_143_);
return v___x_144_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_CtorNames_0__Aesop_CtorNames_nameBase(lean_object* v_x_145_){
_start:
{
switch(lean_obj_tag(v_x_145_))
{
case 0:
{
return v_x_145_;
}
case 1:
{
lean_object* v_str_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v_str_146_ = lean_ctor_get(v_x_145_, 1);
lean_inc_ref(v_str_146_);
lean_dec_ref_known(v_x_145_, 2);
v___x_147_ = lean_box(0);
v___x_148_ = l_Lean_Name_str___override(v___x_147_, v_str_146_);
return v___x_148_;
}
default: 
{
lean_object* v_i_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v_i_149_ = lean_ctor_get(v_x_145_, 1);
lean_inc(v_i_149_);
lean_dec_ref_known(v_x_145_, 2);
v___x_150_ = lean_box(0);
v___x_151_ = l_Lean_Name_num___override(v___x_150_, v_i_149_);
return v___x_151_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toInductionAltLHS_spec__0(size_t v_sz_152_, size_t v_i_153_, lean_object* v_bs_154_){
_start:
{
uint8_t v___x_155_; 
v___x_155_ = lean_usize_dec_lt(v_i_153_, v_sz_152_);
if (v___x_155_ == 0)
{
return v_bs_154_;
}
else
{
lean_object* v_v_156_; lean_object* v___x_157_; lean_object* v_bs_x27_158_; lean_object* v___x_159_; size_t v___x_160_; size_t v___x_161_; lean_object* v___x_162_; 
v_v_156_ = lean_array_uget(v_bs_154_, v_i_153_);
v___x_157_ = lean_unsigned_to_nat(0u);
v_bs_x27_158_ = lean_array_uset(v_bs_154_, v_i_153_, v___x_157_);
v___x_159_ = l_Lean_mkIdent(v_v_156_);
v___x_160_ = ((size_t)1ULL);
v___x_161_ = lean_usize_add(v_i_153_, v___x_160_);
v___x_162_ = lean_array_uset(v_bs_x27_158_, v_i_153_, v___x_159_);
v_i_153_ = v___x_161_;
v_bs_154_ = v___x_162_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toInductionAltLHS_spec__0___boxed(lean_object* v_sz_164_, lean_object* v_i_165_, lean_object* v_bs_166_){
_start:
{
size_t v_sz_boxed_167_; size_t v_i_boxed_168_; lean_object* v_res_169_; 
v_sz_boxed_167_ = lean_unbox_usize(v_sz_164_);
lean_dec(v_sz_164_);
v_i_boxed_168_ = lean_unbox_usize(v_i_165_);
lean_dec(v_i_165_);
v_res_169_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toInductionAltLHS_spec__0(v_sz_boxed_167_, v_i_boxed_168_, v_bs_166_);
return v_res_169_;
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__5(void){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_180_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__2));
v___x_181_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_182_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
lean_ctor_set(v___x_182_, 1, v___x_180_);
return v___x_182_;
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__6(void){
_start:
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v___x_183_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__11);
v___x_184_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_185_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_186_ = l_Lean_Syntax_node1(v___x_185_, v___x_184_, v___x_183_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toInductionAltLHS(lean_object* v_cn_187_){
_start:
{
lean_object* v_ctor_188_; lean_object* v_args_189_; uint8_t v_hasImplicitArg_190_; size_t v_sz_191_; size_t v___x_192_; lean_object* v_ns_193_; lean_object* v___x_194_; lean_object* v_ctor_195_; 
v_ctor_188_ = lean_ctor_get(v_cn_187_, 0);
lean_inc(v_ctor_188_);
v_args_189_ = lean_ctor_get(v_cn_187_, 1);
lean_inc_ref(v_args_189_);
v_hasImplicitArg_190_ = lean_ctor_get_uint8(v_cn_187_, sizeof(void*)*2);
lean_dec_ref(v_cn_187_);
v_sz_191_ = lean_array_size(v_args_189_);
v___x_192_ = ((size_t)0ULL);
v_ns_193_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toInductionAltLHS_spec__0(v_sz_191_, v___x_192_, v_args_189_);
v___x_194_ = lp_aesop___private_Aesop_Script_CtorNames_0__Aesop_CtorNames_nameBase(v_ctor_188_);
v_ctor_195_ = l_Lean_mkIdent(v___x_194_);
if (v_hasImplicitArg_190_ == 0)
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_196_ = lean_box(0);
v___x_197_ = l_Lean_SourceInfo_fromRef(v___x_196_, v_hasImplicitArg_190_);
v___x_198_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1));
v___x_199_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__2));
lean_inc_n(v___x_197_, 4);
v___x_200_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_197_);
lean_ctor_set(v___x_200_, 1, v___x_199_);
v___x_201_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__4));
v___x_202_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_203_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_204_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_204_, 0, v___x_197_);
lean_ctor_set(v___x_204_, 1, v___x_202_);
lean_ctor_set(v___x_204_, 2, v___x_203_);
v___x_205_ = l_Lean_Syntax_node2(v___x_197_, v___x_201_, v___x_204_, v_ctor_195_);
v___x_206_ = l_Array_append___redArg(v___x_203_, v_ns_193_);
lean_dec_ref(v_ns_193_);
v___x_207_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_207_, 0, v___x_197_);
lean_ctor_set(v___x_207_, 1, v___x_202_);
lean_ctor_set(v___x_207_, 2, v___x_206_);
v___x_208_ = l_Lean_Syntax_node3(v___x_197_, v___x_198_, v___x_200_, v___x_205_, v___x_207_);
return v___x_208_;
}
else
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_209_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_210_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__1));
v___x_211_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__5, &lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__5_once, _init_lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__5);
v___x_212_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__4));
v___x_213_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_214_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__6, &lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__6_once, _init_lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__6);
v___x_215_ = l_Lean_Syntax_node2(v___x_209_, v___x_212_, v___x_214_, v_ctor_195_);
v___x_216_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_217_ = l_Array_append___redArg(v___x_216_, v_ns_193_);
lean_dec_ref(v_ns_193_);
v___x_218_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_218_, 0, v___x_209_);
lean_ctor_set(v___x_218_, 1, v___x_213_);
lean_ctor_set(v___x_218_, 2, v___x_217_);
v___x_219_ = l_Lean_Syntax_node3(v___x_209_, v___x_210_, v___x_211_, v___x_215_, v___x_218_);
return v___x_219_;
}
}
}
static lean_object* _init_lp_aesop_Aesop_CtorNames_toInductionAlt___closed__3(void){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_227_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAlt___closed__2));
v___x_228_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_229_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set(v___x_229_, 1, v___x_227_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt(lean_object* v_cn_243_, lean_object* v_tacticSeq_244_){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_245_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_246_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAlt___closed__1));
v___x_247_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_248_ = lp_aesop_Aesop_CtorNames_toInductionAltLHS(v_cn_243_);
v___x_249_ = l_Lean_Syntax_node1(v___x_245_, v___x_247_, v___x_248_);
v___x_250_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toInductionAlt___closed__3, &lp_aesop_Aesop_CtorNames_toInductionAlt___closed__3_once, _init_lp_aesop_Aesop_CtorNames_toInductionAlt___closed__3);
v___x_251_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAlt___closed__5));
v___x_252_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAlt___closed__7));
v___x_253_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_254_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAlt___closed__8));
v___x_255_ = l_Lean_Syntax_SepArray_ofElems(v___x_254_, v_tacticSeq_244_);
v___x_256_ = l_Array_append___redArg(v___x_253_, v___x_255_);
lean_dec_ref(v___x_255_);
v___x_257_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_257_, 0, v___x_245_);
lean_ctor_set(v___x_257_, 1, v___x_247_);
lean_ctor_set(v___x_257_, 2, v___x_256_);
v___x_258_ = l_Lean_Syntax_node1(v___x_245_, v___x_252_, v___x_257_);
v___x_259_ = l_Lean_Syntax_node1(v___x_245_, v___x_251_, v___x_258_);
v___x_260_ = l_Lean_Syntax_node2(v___x_245_, v___x_247_, v___x_250_, v___x_259_);
v___x_261_ = l_Lean_Syntax_node2(v___x_245_, v___x_246_, v___x_249_, v___x_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toInductionAlt___boxed(lean_object* v_cn_262_, lean_object* v_tacticSeq_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_aesop_Aesop_CtorNames_toInductionAlt(v_cn_262_, v_tacticSeq_263_);
lean_dec_ref(v_tacticSeq_263_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_toAltVarNames(lean_object* v_cn_265_){
_start:
{
lean_object* v_args_266_; uint8_t v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; 
v_args_266_ = lean_ctor_get(v_cn_265_, 1);
lean_inc_ref(v_args_266_);
lean_dec_ref(v_cn_265_);
v___x_267_ = 1;
v___x_268_ = lean_array_to_list(v_args_266_);
v___x_269_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_269_, 0, v___x_268_);
lean_ctor_set_uint8(v___x_269_, sizeof(void*)*1, v___x_267_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CtorNames_mkFreshArgNames(lean_object* v_lctx_270_, lean_object* v_cn_271_){
_start:
{
lean_object* v_ctor_272_; lean_object* v_args_273_; uint8_t v_hasImplicitArg_274_; lean_object* v___x_276_; uint8_t v_isShared_277_; uint8_t v_isSharedCheck_291_; 
v_ctor_272_ = lean_ctor_get(v_cn_271_, 0);
v_args_273_ = lean_ctor_get(v_cn_271_, 1);
v_hasImplicitArg_274_ = lean_ctor_get_uint8(v_cn_271_, sizeof(void*)*2);
v_isSharedCheck_291_ = !lean_is_exclusive(v_cn_271_);
if (v_isSharedCheck_291_ == 0)
{
v___x_276_ = v_cn_271_;
v_isShared_277_ = v_isSharedCheck_291_;
goto v_resetjp_275_;
}
else
{
lean_inc(v_args_273_);
lean_inc(v_ctor_272_);
lean_dec(v_cn_271_);
v___x_276_ = lean_box(0);
v_isShared_277_ = v_isSharedCheck_291_;
goto v_resetjp_275_;
}
v_resetjp_275_:
{
lean_object* v___x_278_; lean_object* v_fst_279_; lean_object* v_snd_280_; lean_object* v___x_282_; uint8_t v_isShared_283_; uint8_t v_isSharedCheck_290_; 
v___x_278_ = lp_aesop_Aesop_getUnusedNames(v_lctx_270_, v_args_273_);
lean_dec_ref(v_args_273_);
v_fst_279_ = lean_ctor_get(v___x_278_, 0);
v_snd_280_ = lean_ctor_get(v___x_278_, 1);
v_isSharedCheck_290_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_290_ == 0)
{
v___x_282_ = v___x_278_;
v_isShared_283_ = v_isSharedCheck_290_;
goto v_resetjp_281_;
}
else
{
lean_inc(v_snd_280_);
lean_inc(v_fst_279_);
lean_dec(v___x_278_);
v___x_282_ = lean_box(0);
v_isShared_283_ = v_isSharedCheck_290_;
goto v_resetjp_281_;
}
v_resetjp_281_:
{
lean_object* v___x_285_; 
if (v_isShared_277_ == 0)
{
lean_ctor_set(v___x_276_, 1, v_fst_279_);
v___x_285_ = v___x_276_;
goto v_reusejp_284_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v_ctor_272_);
lean_ctor_set(v_reuseFailAlloc_289_, 1, v_fst_279_);
lean_ctor_set_uint8(v_reuseFailAlloc_289_, sizeof(void*)*2, v_hasImplicitArg_274_);
v___x_285_ = v_reuseFailAlloc_289_;
goto v_reusejp_284_;
}
v_reusejp_284_:
{
lean_object* v___x_287_; 
if (v_isShared_283_ == 0)
{
lean_ctor_set(v___x_282_, 0, v___x_285_);
v___x_287_ = v___x_282_;
goto v_reusejp_286_;
}
else
{
lean_object* v_reuseFailAlloc_288_; 
v_reuseFailAlloc_288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_288_, 0, v___x_285_);
lean_ctor_set(v_reuseFailAlloc_288_, 1, v_snd_280_);
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
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToRCasesPats_spec__0(size_t v_sz_292_, size_t v_i_293_, lean_object* v_bs_294_){
_start:
{
uint8_t v___x_295_; 
v___x_295_ = lean_usize_dec_lt(v_i_293_, v_sz_292_);
if (v___x_295_ == 0)
{
return v_bs_294_;
}
else
{
lean_object* v_v_296_; lean_object* v___x_297_; lean_object* v_bs_x27_298_; lean_object* v___x_299_; size_t v___x_300_; size_t v___x_301_; lean_object* v___x_302_; 
v_v_296_ = lean_array_uget(v_bs_294_, v_i_293_);
v___x_297_ = lean_unsigned_to_nat(0u);
v_bs_x27_298_ = lean_array_uset(v_bs_294_, v_i_293_, v___x_297_);
v___x_299_ = lp_aesop_Aesop_CtorNames_toRCasesPat(v_v_296_);
v___x_300_ = ((size_t)1ULL);
v___x_301_ = lean_usize_add(v_i_293_, v___x_300_);
v___x_302_ = lean_array_uset(v_bs_x27_298_, v_i_293_, v___x_299_);
v_i_293_ = v___x_301_;
v_bs_294_ = v___x_302_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToRCasesPats_spec__0___boxed(lean_object* v_sz_304_, lean_object* v_i_305_, lean_object* v_bs_306_){
_start:
{
size_t v_sz_boxed_307_; size_t v_i_boxed_308_; lean_object* v_res_309_; 
v_sz_boxed_307_ = lean_unbox_usize(v_sz_304_);
lean_dec(v_sz_304_);
v_i_boxed_308_ = lean_unbox_usize(v_i_305_);
lean_dec(v_i_305_);
v_res_309_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToRCasesPats_spec__0(v_sz_boxed_307_, v_i_boxed_308_, v_bs_306_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ctorNamesToRCasesPats(lean_object* v_cns_310_){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; size_t v_sz_316_; size_t v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_311_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_312_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__6));
v___x_313_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_314_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_315_ = ((lean_object*)(lp_aesop_Aesop_CtorNames_toInductionAltLHS___closed__2));
v_sz_316_ = lean_array_size(v_cns_310_);
v___x_317_ = ((size_t)0ULL);
v___x_318_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToRCasesPats_spec__0(v_sz_316_, v___x_317_, v_cns_310_);
v___x_319_ = l_Lean_Syntax_SepArray_ofElems(v___x_315_, v___x_318_);
lean_dec_ref(v___x_318_);
v___x_320_ = l_Array_append___redArg(v___x_314_, v___x_319_);
lean_dec_ref(v___x_319_);
v___x_321_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_321_, 0, v___x_311_);
lean_ctor_set(v___x_321_, 1, v___x_313_);
lean_ctor_set(v___x_321_, 2, v___x_320_);
v___x_322_ = l_Lean_Syntax_node1(v___x_311_, v___x_312_, v___x_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToInductionAlts_spec__0(size_t v_sz_323_, size_t v_i_324_, lean_object* v_bs_325_){
_start:
{
uint8_t v___x_326_; 
v___x_326_ = lean_usize_dec_lt(v_i_324_, v_sz_323_);
if (v___x_326_ == 0)
{
return v_bs_325_;
}
else
{
lean_object* v_v_327_; lean_object* v_fst_328_; lean_object* v_snd_329_; lean_object* v___x_330_; lean_object* v_bs_x27_331_; lean_object* v___x_332_; size_t v___x_333_; size_t v___x_334_; lean_object* v___x_335_; 
v_v_327_ = lean_array_uget_borrowed(v_bs_325_, v_i_324_);
v_fst_328_ = lean_ctor_get(v_v_327_, 0);
lean_inc(v_fst_328_);
v_snd_329_ = lean_ctor_get(v_v_327_, 1);
lean_inc(v_snd_329_);
v___x_330_ = lean_unsigned_to_nat(0u);
v_bs_x27_331_ = lean_array_uset(v_bs_325_, v_i_324_, v___x_330_);
v___x_332_ = lp_aesop_Aesop_CtorNames_toInductionAlt(v_fst_328_, v_snd_329_);
lean_dec(v_snd_329_);
v___x_333_ = ((size_t)1ULL);
v___x_334_ = lean_usize_add(v_i_324_, v___x_333_);
v___x_335_ = lean_array_uset(v_bs_x27_331_, v_i_324_, v___x_332_);
v_i_324_ = v___x_334_;
v_bs_325_ = v___x_335_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToInductionAlts_spec__0___boxed(lean_object* v_sz_337_, lean_object* v_i_338_, lean_object* v_bs_339_){
_start:
{
size_t v_sz_boxed_340_; size_t v_i_boxed_341_; lean_object* v_res_342_; 
v_sz_boxed_340_ = lean_unbox_usize(v_sz_337_);
lean_dec(v_sz_337_);
v_i_boxed_341_ = lean_unbox_usize(v_i_338_);
lean_dec(v_i_338_);
v_res_342_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToInductionAlts_spec__0(v_sz_boxed_340_, v_i_boxed_341_, v_bs_339_);
return v_res_342_;
}
}
static lean_object* _init_lp_aesop_Aesop_ctorNamesToInductionAlts___closed__3(void){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_350_ = ((lean_object*)(lp_aesop_Aesop_ctorNamesToInductionAlts___closed__2));
v___x_351_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_352_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
lean_ctor_set(v___x_352_, 1, v___x_350_);
return v___x_352_;
}
}
static lean_object* _init_lp_aesop_Aesop_ctorNamesToInductionAlts___closed__4(void){
_start:
{
lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_353_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_354_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_355_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_356_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_356_, 0, v___x_355_);
lean_ctor_set(v___x_356_, 1, v___x_354_);
lean_ctor_set(v___x_356_, 2, v___x_353_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ctorNamesToInductionAlts(lean_object* v_cns_357_){
_start:
{
size_t v_sz_358_; size_t v___x_359_; lean_object* v_alts_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
v_sz_358_ = lean_array_size(v_cns_357_);
v___x_359_ = ((size_t)0ULL);
v_alts_360_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ctorNamesToInductionAlts_spec__0(v_sz_358_, v___x_359_, v_cns_357_);
v___x_361_ = lean_obj_once(&lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7, &lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7_once, _init_lp_aesop_Aesop_CtorNames_toRCasesPat___closed__7);
v___x_362_ = ((lean_object*)(lp_aesop_Aesop_ctorNamesToInductionAlts___closed__1));
v___x_363_ = lean_obj_once(&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__3, &lp_aesop_Aesop_ctorNamesToInductionAlts___closed__3_once, _init_lp_aesop_Aesop_ctorNamesToInductionAlts___closed__3);
v___x_364_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__8));
v___x_365_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CtorNames_toRCasesPat_spec__0___closed__12);
v___x_366_ = lean_obj_once(&lp_aesop_Aesop_ctorNamesToInductionAlts___closed__4, &lp_aesop_Aesop_ctorNamesToInductionAlts___closed__4_once, _init_lp_aesop_Aesop_ctorNamesToInductionAlts___closed__4);
v___x_367_ = l_Array_append___redArg(v___x_365_, v_alts_360_);
lean_dec_ref(v_alts_360_);
v___x_368_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_368_, 0, v___x_361_);
lean_ctor_set(v___x_368_, 1, v___x_364_);
lean_ctor_set(v___x_368_, 2, v___x_367_);
v___x_369_ = l_Lean_Syntax_node3(v___x_361_, v___x_362_, v___x_363_, v___x_366_, v___x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg___lam__0(lean_object* v_k_370_, lean_object* v_b_371_, lean_object* v_c_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_){
_start:
{
lean_object* v___x_378_; 
lean_inc(v___y_376_);
lean_inc_ref(v___y_375_);
lean_inc(v___y_374_);
lean_inc_ref(v___y_373_);
v___x_378_ = lean_apply_7(v_k_370_, v_b_371_, v_c_372_, v___y_373_, v___y_374_, v___y_375_, v___y_376_, lean_box(0));
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg___lam__0___boxed(lean_object* v_k_379_, lean_object* v_b_380_, lean_object* v_c_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg___lam__0(v_k_379_, v_b_380_, v_c_381_, v___y_382_, v___y_383_, v___y_384_, v___y_385_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg(lean_object* v_type_388_, lean_object* v_maxFVars_x3f_389_, lean_object* v_k_390_, uint8_t v_cleanupAnnotations_391_, uint8_t v_whnfType_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_){
_start:
{
lean_object* v___f_398_; lean_object* v___x_399_; 
v___f_398_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_398_, 0, v_k_390_);
v___x_399_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_box(0), v_type_388_, v_maxFVars_x3f_389_, v___f_398_, v_cleanupAnnotations_391_, v_whnfType_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_399_) == 0)
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
v_a_400_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_399_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_399_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
lean_object* v___x_405_; 
if (v_isShared_403_ == 0)
{
v___x_405_ = v___x_402_;
goto v_reusejp_404_;
}
else
{
lean_object* v_reuseFailAlloc_406_; 
v_reuseFailAlloc_406_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_406_, 0, v_a_400_);
v___x_405_ = v_reuseFailAlloc_406_;
goto v_reusejp_404_;
}
v_reusejp_404_:
{
return v___x_405_;
}
}
}
else
{
lean_object* v_a_408_; lean_object* v___x_410_; uint8_t v_isShared_411_; uint8_t v_isSharedCheck_415_; 
v_a_408_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_415_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_415_ == 0)
{
v___x_410_ = v___x_399_;
v_isShared_411_ = v_isSharedCheck_415_;
goto v_resetjp_409_;
}
else
{
lean_inc(v_a_408_);
lean_dec(v___x_399_);
v___x_410_ = lean_box(0);
v_isShared_411_ = v_isSharedCheck_415_;
goto v_resetjp_409_;
}
v_resetjp_409_:
{
lean_object* v___x_413_; 
if (v_isShared_411_ == 0)
{
v___x_413_ = v___x_410_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_a_408_);
v___x_413_ = v_reuseFailAlloc_414_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
return v___x_413_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg___boxed(lean_object* v_type_416_, lean_object* v_maxFVars_x3f_417_, lean_object* v_k_418_, lean_object* v_cleanupAnnotations_419_, lean_object* v_whnfType_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_426_; uint8_t v_whnfType_boxed_427_; lean_object* v_res_428_; 
v_cleanupAnnotations_boxed_426_ = lean_unbox(v_cleanupAnnotations_419_);
v_whnfType_boxed_427_ = lean_unbox(v_whnfType_420_);
v_res_428_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg(v_type_416_, v_maxFVars_x3f_417_, v_k_418_, v_cleanupAnnotations_boxed_426_, v_whnfType_boxed_427_, v___y_421_, v___y_422_, v___y_423_, v___y_424_);
lean_dec(v___y_424_);
lean_dec_ref(v___y_423_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2(lean_object* v_00_u03b1_429_, lean_object* v_type_430_, lean_object* v_maxFVars_x3f_431_, lean_object* v_k_432_, uint8_t v_cleanupAnnotations_433_, uint8_t v_whnfType_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_){
_start:
{
lean_object* v___x_440_; 
v___x_440_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg(v_type_430_, v_maxFVars_x3f_431_, v_k_432_, v_cleanupAnnotations_433_, v_whnfType_434_, v___y_435_, v___y_436_, v___y_437_, v___y_438_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___boxed(lean_object* v_00_u03b1_441_, lean_object* v_type_442_, lean_object* v_maxFVars_x3f_443_, lean_object* v_k_444_, lean_object* v_cleanupAnnotations_445_, lean_object* v_whnfType_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_452_; uint8_t v_whnfType_boxed_453_; lean_object* v_res_454_; 
v_cleanupAnnotations_boxed_452_ = lean_unbox(v_cleanupAnnotations_445_);
v_whnfType_boxed_453_ = lean_unbox(v_whnfType_446_);
v_res_454_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2(v_00_u03b1_441_, v_type_442_, v_maxFVars_x3f_443_, v_k_444_, v_cleanupAnnotations_boxed_452_, v_whnfType_boxed_453_, v___y_447_, v___y_448_, v___y_449_, v___y_450_);
lean_dec(v___y_450_);
lean_dec_ref(v___y_449_);
lean_dec(v___y_448_);
lean_dec_ref(v___y_447_);
return v_res_454_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0_spec__2(lean_object* v_msgData_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_, lean_object* v___y_459_){
_start:
{
lean_object* v___x_461_; lean_object* v_env_462_; lean_object* v___x_463_; lean_object* v_mctx_464_; lean_object* v_lctx_465_; lean_object* v_options_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_461_ = lean_st_ref_get(v___y_459_);
v_env_462_ = lean_ctor_get(v___x_461_, 0);
lean_inc_ref(v_env_462_);
lean_dec(v___x_461_);
v___x_463_ = lean_st_ref_get(v___y_457_);
v_mctx_464_ = lean_ctor_get(v___x_463_, 0);
lean_inc_ref(v_mctx_464_);
lean_dec(v___x_463_);
v_lctx_465_ = lean_ctor_get(v___y_456_, 2);
v_options_466_ = lean_ctor_get(v___y_458_, 2);
lean_inc_ref(v_options_466_);
lean_inc_ref(v_lctx_465_);
v___x_467_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_467_, 0, v_env_462_);
lean_ctor_set(v___x_467_, 1, v_mctx_464_);
lean_ctor_set(v___x_467_, 2, v_lctx_465_);
lean_ctor_set(v___x_467_, 3, v_options_466_);
v___x_468_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_468_, 0, v___x_467_);
lean_ctor_set(v___x_468_, 1, v_msgData_455_);
v___x_469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_469_, 0, v___x_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0_spec__2___boxed(lean_object* v_msgData_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0_spec__2(v_msgData_470_, v___y_471_, v___y_472_, v___y_473_, v___y_474_);
lean_dec(v___y_474_);
lean_dec_ref(v___y_473_);
lean_dec(v___y_472_);
lean_dec_ref(v___y_471_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___redArg(lean_object* v_msg_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
lean_object* v_ref_483_; lean_object* v___x_484_; lean_object* v_a_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_493_; 
v_ref_483_ = lean_ctor_get(v___y_480_, 5);
v___x_484_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0_spec__2(v_msg_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
v_a_485_ = lean_ctor_get(v___x_484_, 0);
v_isSharedCheck_493_ = !lean_is_exclusive(v___x_484_);
if (v_isSharedCheck_493_ == 0)
{
v___x_487_ = v___x_484_;
v_isShared_488_ = v_isSharedCheck_493_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_a_485_);
lean_dec(v___x_484_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_493_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v___x_489_; lean_object* v___x_491_; 
lean_inc(v_ref_483_);
v___x_489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_489_, 0, v_ref_483_);
lean_ctor_set(v___x_489_, 1, v_a_485_);
if (v_isShared_488_ == 0)
{
lean_ctor_set_tag(v___x_487_, 1);
lean_ctor_set(v___x_487_, 0, v___x_489_);
v___x_491_ = v___x_487_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_492_; 
v_reuseFailAlloc_492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_492_, 0, v___x_489_);
v___x_491_ = v_reuseFailAlloc_492_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
return v___x_491_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___redArg___boxed(lean_object* v_msg_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_){
_start:
{
lean_object* v_res_500_; 
v_res_500_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___redArg(v_msg_494_, v___y_495_, v___y_496_, v___y_497_, v___y_498_);
lean_dec(v___y_498_);
lean_dec_ref(v___y_497_);
lean_dec(v___y_496_);
lean_dec_ref(v___y_495_);
return v_res_500_;
}
}
static lean_object* _init_lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__0(void){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = l_instMonadEIO(lean_box(0));
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1(lean_object* v_msg_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_){
_start:
{
lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v_toApplicative_514_; lean_object* v___x_516_; uint8_t v_isShared_517_; uint8_t v_isSharedCheck_575_; 
v___x_512_ = lean_obj_once(&lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__0, &lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__0_once, _init_lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__0);
v___x_513_ = l_StateRefT_x27_instMonad___redArg(v___x_512_);
v_toApplicative_514_ = lean_ctor_get(v___x_513_, 0);
v_isSharedCheck_575_ = !lean_is_exclusive(v___x_513_);
if (v_isSharedCheck_575_ == 0)
{
lean_object* v_unused_576_; 
v_unused_576_ = lean_ctor_get(v___x_513_, 1);
lean_dec(v_unused_576_);
v___x_516_ = v___x_513_;
v_isShared_517_ = v_isSharedCheck_575_;
goto v_resetjp_515_;
}
else
{
lean_inc(v_toApplicative_514_);
lean_dec(v___x_513_);
v___x_516_ = lean_box(0);
v_isShared_517_ = v_isSharedCheck_575_;
goto v_resetjp_515_;
}
v_resetjp_515_:
{
lean_object* v_toFunctor_518_; lean_object* v_toSeq_519_; lean_object* v_toSeqLeft_520_; lean_object* v_toSeqRight_521_; lean_object* v___x_523_; uint8_t v_isShared_524_; uint8_t v_isSharedCheck_573_; 
v_toFunctor_518_ = lean_ctor_get(v_toApplicative_514_, 0);
v_toSeq_519_ = lean_ctor_get(v_toApplicative_514_, 2);
v_toSeqLeft_520_ = lean_ctor_get(v_toApplicative_514_, 3);
v_toSeqRight_521_ = lean_ctor_get(v_toApplicative_514_, 4);
v_isSharedCheck_573_ = !lean_is_exclusive(v_toApplicative_514_);
if (v_isSharedCheck_573_ == 0)
{
lean_object* v_unused_574_; 
v_unused_574_ = lean_ctor_get(v_toApplicative_514_, 1);
lean_dec(v_unused_574_);
v___x_523_ = v_toApplicative_514_;
v_isShared_524_ = v_isSharedCheck_573_;
goto v_resetjp_522_;
}
else
{
lean_inc(v_toSeqRight_521_);
lean_inc(v_toSeqLeft_520_);
lean_inc(v_toSeq_519_);
lean_inc(v_toFunctor_518_);
lean_dec(v_toApplicative_514_);
v___x_523_ = lean_box(0);
v_isShared_524_ = v_isSharedCheck_573_;
goto v_resetjp_522_;
}
v_resetjp_522_:
{
lean_object* v___f_525_; lean_object* v___f_526_; lean_object* v___f_527_; lean_object* v___f_528_; lean_object* v___x_529_; lean_object* v___f_530_; lean_object* v___f_531_; lean_object* v___f_532_; lean_object* v___x_534_; 
v___f_525_ = ((lean_object*)(lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__1));
v___f_526_ = ((lean_object*)(lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__2));
lean_inc_ref(v_toFunctor_518_);
v___f_527_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_527_, 0, v_toFunctor_518_);
v___f_528_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_528_, 0, v_toFunctor_518_);
v___x_529_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_529_, 0, v___f_527_);
lean_ctor_set(v___x_529_, 1, v___f_528_);
v___f_530_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_530_, 0, v_toSeqRight_521_);
v___f_531_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_531_, 0, v_toSeqLeft_520_);
v___f_532_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_532_, 0, v_toSeq_519_);
if (v_isShared_524_ == 0)
{
lean_ctor_set(v___x_523_, 4, v___f_530_);
lean_ctor_set(v___x_523_, 3, v___f_531_);
lean_ctor_set(v___x_523_, 2, v___f_532_);
lean_ctor_set(v___x_523_, 1, v___f_525_);
lean_ctor_set(v___x_523_, 0, v___x_529_);
v___x_534_ = v___x_523_;
goto v_reusejp_533_;
}
else
{
lean_object* v_reuseFailAlloc_572_; 
v_reuseFailAlloc_572_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_572_, 0, v___x_529_);
lean_ctor_set(v_reuseFailAlloc_572_, 1, v___f_525_);
lean_ctor_set(v_reuseFailAlloc_572_, 2, v___f_532_);
lean_ctor_set(v_reuseFailAlloc_572_, 3, v___f_531_);
lean_ctor_set(v_reuseFailAlloc_572_, 4, v___f_530_);
v___x_534_ = v_reuseFailAlloc_572_;
goto v_reusejp_533_;
}
v_reusejp_533_:
{
lean_object* v___x_536_; 
if (v_isShared_517_ == 0)
{
lean_ctor_set(v___x_516_, 1, v___f_526_);
lean_ctor_set(v___x_516_, 0, v___x_534_);
v___x_536_ = v___x_516_;
goto v_reusejp_535_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v___x_534_);
lean_ctor_set(v_reuseFailAlloc_571_, 1, v___f_526_);
v___x_536_ = v_reuseFailAlloc_571_;
goto v_reusejp_535_;
}
v_reusejp_535_:
{
lean_object* v___x_537_; lean_object* v_toApplicative_538_; lean_object* v___x_540_; uint8_t v_isShared_541_; uint8_t v_isSharedCheck_569_; 
v___x_537_ = l_StateRefT_x27_instMonad___redArg(v___x_536_);
v_toApplicative_538_ = lean_ctor_get(v___x_537_, 0);
v_isSharedCheck_569_ = !lean_is_exclusive(v___x_537_);
if (v_isSharedCheck_569_ == 0)
{
lean_object* v_unused_570_; 
v_unused_570_ = lean_ctor_get(v___x_537_, 1);
lean_dec(v_unused_570_);
v___x_540_ = v___x_537_;
v_isShared_541_ = v_isSharedCheck_569_;
goto v_resetjp_539_;
}
else
{
lean_inc(v_toApplicative_538_);
lean_dec(v___x_537_);
v___x_540_ = lean_box(0);
v_isShared_541_ = v_isSharedCheck_569_;
goto v_resetjp_539_;
}
v_resetjp_539_:
{
lean_object* v_toFunctor_542_; lean_object* v_toSeq_543_; lean_object* v_toSeqLeft_544_; lean_object* v_toSeqRight_545_; lean_object* v___x_547_; uint8_t v_isShared_548_; uint8_t v_isSharedCheck_567_; 
v_toFunctor_542_ = lean_ctor_get(v_toApplicative_538_, 0);
v_toSeq_543_ = lean_ctor_get(v_toApplicative_538_, 2);
v_toSeqLeft_544_ = lean_ctor_get(v_toApplicative_538_, 3);
v_toSeqRight_545_ = lean_ctor_get(v_toApplicative_538_, 4);
v_isSharedCheck_567_ = !lean_is_exclusive(v_toApplicative_538_);
if (v_isSharedCheck_567_ == 0)
{
lean_object* v_unused_568_; 
v_unused_568_ = lean_ctor_get(v_toApplicative_538_, 1);
lean_dec(v_unused_568_);
v___x_547_ = v_toApplicative_538_;
v_isShared_548_ = v_isSharedCheck_567_;
goto v_resetjp_546_;
}
else
{
lean_inc(v_toSeqRight_545_);
lean_inc(v_toSeqLeft_544_);
lean_inc(v_toSeq_543_);
lean_inc(v_toFunctor_542_);
lean_dec(v_toApplicative_538_);
v___x_547_ = lean_box(0);
v_isShared_548_ = v_isSharedCheck_567_;
goto v_resetjp_546_;
}
v_resetjp_546_:
{
lean_object* v___f_549_; lean_object* v___f_550_; lean_object* v___f_551_; lean_object* v___f_552_; lean_object* v___x_553_; lean_object* v___f_554_; lean_object* v___f_555_; lean_object* v___f_556_; lean_object* v___x_558_; 
v___f_549_ = ((lean_object*)(lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__3));
v___f_550_ = ((lean_object*)(lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___closed__4));
lean_inc_ref(v_toFunctor_542_);
v___f_551_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_551_, 0, v_toFunctor_542_);
v___f_552_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_552_, 0, v_toFunctor_542_);
v___x_553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_553_, 0, v___f_551_);
lean_ctor_set(v___x_553_, 1, v___f_552_);
v___f_554_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_554_, 0, v_toSeqRight_545_);
v___f_555_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_555_, 0, v_toSeqLeft_544_);
v___f_556_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_556_, 0, v_toSeq_543_);
if (v_isShared_548_ == 0)
{
lean_ctor_set(v___x_547_, 4, v___f_554_);
lean_ctor_set(v___x_547_, 3, v___f_555_);
lean_ctor_set(v___x_547_, 2, v___f_556_);
lean_ctor_set(v___x_547_, 1, v___f_549_);
lean_ctor_set(v___x_547_, 0, v___x_553_);
v___x_558_ = v___x_547_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_566_; 
v_reuseFailAlloc_566_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_566_, 0, v___x_553_);
lean_ctor_set(v_reuseFailAlloc_566_, 1, v___f_549_);
lean_ctor_set(v_reuseFailAlloc_566_, 2, v___f_556_);
lean_ctor_set(v_reuseFailAlloc_566_, 3, v___f_555_);
lean_ctor_set(v_reuseFailAlloc_566_, 4, v___f_554_);
v___x_558_ = v_reuseFailAlloc_566_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
lean_object* v___x_560_; 
if (v_isShared_541_ == 0)
{
lean_ctor_set(v___x_540_, 1, v___f_550_);
lean_ctor_set(v___x_540_, 0, v___x_558_);
v___x_560_ = v___x_540_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v___x_558_);
lean_ctor_set(v_reuseFailAlloc_565_, 1, v___f_550_);
v___x_560_ = v_reuseFailAlloc_565_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_3364__overap_563_; lean_object* v___x_564_; 
v___x_561_ = lean_box(0);
v___x_562_ = l_instInhabitedOfMonad___redArg(v___x_560_, v___x_561_);
v___x_3364__overap_563_ = lean_panic_fn_borrowed(v___x_562_, v_msg_506_);
lean_dec(v___x_562_);
lean_inc(v___y_510_);
lean_inc_ref(v___y_509_);
lean_inc(v___y_508_);
lean_inc_ref(v___y_507_);
v___x_564_ = lean_apply_5(v___x_3364__overap_563_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, lean_box(0));
return v___x_564_;
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1___boxed(lean_object* v_msg_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1(v_msg_577_, v___y_578_, v___y_579_, v___y_580_, v___y_581_);
lean_dec(v___y_581_);
lean_dec_ref(v___y_580_);
lean_dec(v___y_579_);
lean_dec_ref(v___y_578_);
return v_res_583_;
}
}
static lean_object* _init_lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__1(void){
_start:
{
lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_585_ = ((lean_object*)(lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__0));
v___x_586_ = l_Lean_stringToMessageData(v___x_585_);
return v___x_586_;
}
}
static lean_object* _init_lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__3(void){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; 
v___x_588_ = ((lean_object*)(lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__2));
v___x_589_ = l_Lean_stringToMessageData(v___x_588_);
return v___x_589_;
}
}
static lean_object* _init_lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__7(void){
_start:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; 
v___x_593_ = ((lean_object*)(lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__6));
v___x_594_ = lean_unsigned_to_nat(11u);
v___x_595_ = lean_unsigned_to_nat(122u);
v___x_596_ = ((lean_object*)(lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__5));
v___x_597_ = ((lean_object*)(lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__4));
v___x_598_ = l_mkPanicMessageWithDecl(v___x_597_, v___x_596_, v___x_595_, v___x_594_, v___x_593_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0(lean_object* v_constName_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_){
_start:
{
lean_object* v___x_613_; lean_object* v_env_614_; uint8_t v___x_615_; lean_object* v___x_616_; 
v___x_613_ = lean_st_ref_get(v___y_603_);
v_env_614_ = lean_ctor_get(v___x_613_, 0);
lean_inc_ref(v_env_614_);
lean_dec(v___x_613_);
v___x_615_ = 0;
lean_inc(v_constName_599_);
v___x_616_ = l_Lean_Environment_findAsync_x3f(v_env_614_, v_constName_599_, v___x_615_);
if (lean_obj_tag(v___x_616_) == 1)
{
lean_object* v_val_617_; uint8_t v_kind_618_; 
v_val_617_ = lean_ctor_get(v___x_616_, 0);
lean_inc(v_val_617_);
lean_dec_ref_known(v___x_616_, 1);
v_kind_618_ = lean_ctor_get_uint8(v_val_617_, sizeof(void*)*3);
if (v_kind_618_ == 6)
{
lean_object* v___x_619_; 
v___x_619_ = l_Lean_AsyncConstantInfo_toConstantInfo(v_val_617_);
if (lean_obj_tag(v___x_619_) == 6)
{
lean_object* v_val_620_; lean_object* v___x_622_; uint8_t v_isShared_623_; uint8_t v_isSharedCheck_627_; 
lean_dec(v_constName_599_);
v_val_620_ = lean_ctor_get(v___x_619_, 0);
v_isSharedCheck_627_ = !lean_is_exclusive(v___x_619_);
if (v_isSharedCheck_627_ == 0)
{
v___x_622_ = v___x_619_;
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
else
{
lean_inc(v_val_620_);
lean_dec(v___x_619_);
v___x_622_ = lean_box(0);
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
v_resetjp_621_:
{
lean_object* v___x_625_; 
if (v_isShared_623_ == 0)
{
lean_ctor_set_tag(v___x_622_, 0);
v___x_625_ = v___x_622_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v_val_620_);
v___x_625_ = v_reuseFailAlloc_626_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
return v___x_625_;
}
}
}
else
{
lean_object* v___x_628_; lean_object* v___x_629_; 
lean_dec_ref(v___x_619_);
v___x_628_ = lean_obj_once(&lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__7, &lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__7_once, _init_lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__7);
v___x_629_ = lp_aesop_panic___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__1(v___x_628_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
if (lean_obj_tag(v___x_629_) == 0)
{
lean_object* v_a_630_; lean_object* v___x_632_; uint8_t v_isShared_633_; uint8_t v_isSharedCheck_638_; 
v_a_630_ = lean_ctor_get(v___x_629_, 0);
v_isSharedCheck_638_ = !lean_is_exclusive(v___x_629_);
if (v_isSharedCheck_638_ == 0)
{
v___x_632_ = v___x_629_;
v_isShared_633_ = v_isSharedCheck_638_;
goto v_resetjp_631_;
}
else
{
lean_inc(v_a_630_);
lean_dec(v___x_629_);
v___x_632_ = lean_box(0);
v_isShared_633_ = v_isSharedCheck_638_;
goto v_resetjp_631_;
}
v_resetjp_631_:
{
if (lean_obj_tag(v_a_630_) == 0)
{
lean_del_object(v___x_632_);
goto v___jp_605_;
}
else
{
lean_object* v_val_634_; lean_object* v___x_636_; 
lean_dec(v_constName_599_);
v_val_634_ = lean_ctor_get(v_a_630_, 0);
lean_inc(v_val_634_);
lean_dec_ref_known(v_a_630_, 1);
if (v_isShared_633_ == 0)
{
lean_ctor_set(v___x_632_, 0, v_val_634_);
v___x_636_ = v___x_632_;
goto v_reusejp_635_;
}
else
{
lean_object* v_reuseFailAlloc_637_; 
v_reuseFailAlloc_637_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_637_, 0, v_val_634_);
v___x_636_ = v_reuseFailAlloc_637_;
goto v_reusejp_635_;
}
v_reusejp_635_:
{
return v___x_636_;
}
}
}
}
else
{
lean_object* v_a_639_; lean_object* v___x_641_; uint8_t v_isShared_642_; uint8_t v_isSharedCheck_646_; 
lean_dec(v_constName_599_);
v_a_639_ = lean_ctor_get(v___x_629_, 0);
v_isSharedCheck_646_ = !lean_is_exclusive(v___x_629_);
if (v_isSharedCheck_646_ == 0)
{
v___x_641_ = v___x_629_;
v_isShared_642_ = v_isSharedCheck_646_;
goto v_resetjp_640_;
}
else
{
lean_inc(v_a_639_);
lean_dec(v___x_629_);
v___x_641_ = lean_box(0);
v_isShared_642_ = v_isSharedCheck_646_;
goto v_resetjp_640_;
}
v_resetjp_640_:
{
lean_object* v___x_644_; 
if (v_isShared_642_ == 0)
{
v___x_644_ = v___x_641_;
goto v_reusejp_643_;
}
else
{
lean_object* v_reuseFailAlloc_645_; 
v_reuseFailAlloc_645_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_645_, 0, v_a_639_);
v___x_644_ = v_reuseFailAlloc_645_;
goto v_reusejp_643_;
}
v_reusejp_643_:
{
return v___x_644_;
}
}
}
}
}
else
{
lean_dec(v_val_617_);
goto v___jp_605_;
}
}
else
{
lean_dec(v___x_616_);
goto v___jp_605_;
}
v___jp_605_:
{
lean_object* v___x_606_; uint8_t v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; 
v___x_606_ = lean_obj_once(&lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__1, &lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__1_once, _init_lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__1);
v___x_607_ = 0;
v___x_608_ = l_Lean_MessageData_ofConstName(v_constName_599_, v___x_607_);
v___x_609_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_609_, 0, v___x_606_);
lean_ctor_set(v___x_609_, 1, v___x_608_);
v___x_610_ = lean_obj_once(&lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__3, &lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__3_once, _init_lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___closed__3);
v___x_611_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_609_);
lean_ctor_set(v___x_611_, 1, v___x_610_);
v___x_612_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___redArg(v___x_611_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
return v___x_612_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0___boxed(lean_object* v_constName_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_){
_start:
{
lean_object* v_res_653_; 
v_res_653_ = lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0(v_constName_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
return v_res_653_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___redArg(lean_object* v_a_654_, lean_object* v_b_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_){
_start:
{
lean_object* v_array_660_; lean_object* v_start_661_; lean_object* v_stop_662_; lean_object* v___x_664_; uint8_t v_isShared_665_; uint8_t v_isSharedCheck_705_; 
v_array_660_ = lean_ctor_get(v_a_654_, 0);
v_start_661_ = lean_ctor_get(v_a_654_, 1);
v_stop_662_ = lean_ctor_get(v_a_654_, 2);
v_isSharedCheck_705_ = !lean_is_exclusive(v_a_654_);
if (v_isSharedCheck_705_ == 0)
{
v___x_664_ = v_a_654_;
v_isShared_665_ = v_isSharedCheck_705_;
goto v_resetjp_663_;
}
else
{
lean_inc(v_stop_662_);
lean_inc(v_start_661_);
lean_inc(v_array_660_);
lean_dec(v_a_654_);
v___x_664_ = lean_box(0);
v_isShared_665_ = v_isSharedCheck_705_;
goto v_resetjp_663_;
}
v_resetjp_663_:
{
uint8_t v___x_666_; 
v___x_666_ = lean_nat_dec_lt(v_start_661_, v_stop_662_);
if (v___x_666_ == 0)
{
lean_object* v___x_667_; 
lean_del_object(v___x_664_);
lean_dec(v_stop_662_);
lean_dec(v_start_661_);
lean_dec_ref(v_array_660_);
v___x_667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_667_, 0, v_b_655_);
return v___x_667_;
}
else
{
lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
v___x_668_ = lean_array_fget_borrowed(v_array_660_, v_start_661_);
v___x_669_ = l_Lean_Expr_fvarId_x21(v___x_668_);
v___x_670_ = l_Lean_FVarId_getDecl___redArg(v___x_669_, v___y_656_, v___y_657_, v___y_658_);
if (lean_obj_tag(v___x_670_) == 0)
{
lean_object* v_a_671_; lean_object* v_fst_672_; lean_object* v_snd_673_; lean_object* v___x_675_; uint8_t v_isShared_676_; uint8_t v_isSharedCheck_696_; 
v_a_671_ = lean_ctor_get(v___x_670_, 0);
lean_inc(v_a_671_);
lean_dec_ref_known(v___x_670_, 1);
v_fst_672_ = lean_ctor_get(v_b_655_, 0);
v_snd_673_ = lean_ctor_get(v_b_655_, 1);
v_isSharedCheck_696_ = !lean_is_exclusive(v_b_655_);
if (v_isSharedCheck_696_ == 0)
{
v___x_675_ = v_b_655_;
v_isShared_676_ = v_isSharedCheck_696_;
goto v_resetjp_674_;
}
else
{
lean_inc(v_snd_673_);
lean_inc(v_fst_672_);
lean_dec(v_b_655_);
v___x_675_ = lean_box(0);
v_isShared_676_ = v_isSharedCheck_696_;
goto v_resetjp_674_;
}
v_resetjp_674_:
{
lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_680_; 
v___x_677_ = lean_unsigned_to_nat(1u);
v___x_678_ = lean_nat_add(v_start_661_, v___x_677_);
lean_dec(v_start_661_);
if (v_isShared_665_ == 0)
{
lean_ctor_set(v___x_664_, 1, v___x_678_);
v___x_680_ = v___x_664_;
goto v_reusejp_679_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v_array_660_);
lean_ctor_set(v_reuseFailAlloc_695_, 1, v___x_678_);
lean_ctor_set(v_reuseFailAlloc_695_, 2, v_stop_662_);
v___x_680_ = v_reuseFailAlloc_695_;
goto v_reusejp_679_;
}
v_reusejp_679_:
{
lean_object* v___x_681_; lean_object* v___x_682_; uint8_t v___y_684_; uint8_t v___x_690_; 
v___x_681_ = l_Lean_LocalDecl_userName(v_a_671_);
v___x_682_ = lean_array_push(v_fst_672_, v___x_681_);
v___x_690_ = lean_unbox(v_snd_673_);
if (v___x_690_ == 0)
{
uint8_t v___x_691_; uint8_t v___x_692_; 
v___x_691_ = l_Lean_LocalDecl_binderInfo(v_a_671_);
lean_dec(v_a_671_);
v___x_692_ = l_Lean_BinderInfo_isExplicit(v___x_691_);
if (v___x_692_ == 0)
{
lean_dec(v_snd_673_);
v___y_684_ = v___x_666_;
goto v___jp_683_;
}
else
{
uint8_t v___x_693_; 
v___x_693_ = lean_unbox(v_snd_673_);
lean_dec(v_snd_673_);
v___y_684_ = v___x_693_;
goto v___jp_683_;
}
}
else
{
uint8_t v___x_694_; 
lean_dec(v_a_671_);
v___x_694_ = lean_unbox(v_snd_673_);
lean_dec(v_snd_673_);
v___y_684_ = v___x_694_;
goto v___jp_683_;
}
v___jp_683_:
{
lean_object* v___x_685_; lean_object* v___x_687_; 
v___x_685_ = lean_box(v___y_684_);
if (v_isShared_676_ == 0)
{
lean_ctor_set(v___x_675_, 1, v___x_685_);
lean_ctor_set(v___x_675_, 0, v___x_682_);
v___x_687_ = v___x_675_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_689_; 
v_reuseFailAlloc_689_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_689_, 0, v___x_682_);
lean_ctor_set(v_reuseFailAlloc_689_, 1, v___x_685_);
v___x_687_ = v_reuseFailAlloc_689_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
v_a_654_ = v___x_680_;
v_b_655_ = v___x_687_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_704_; 
lean_del_object(v___x_664_);
lean_dec(v_stop_662_);
lean_dec(v_start_661_);
lean_dec_ref(v_array_660_);
lean_dec_ref(v_b_655_);
v_a_697_ = lean_ctor_get(v___x_670_, 0);
v_isSharedCheck_704_ = !lean_is_exclusive(v___x_670_);
if (v_isSharedCheck_704_ == 0)
{
v___x_699_ = v___x_670_;
v_isShared_700_ = v_isSharedCheck_704_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_a_697_);
lean_dec(v___x_670_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_704_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v___x_702_; 
if (v_isShared_700_ == 0)
{
v___x_702_ = v___x_699_;
goto v_reusejp_701_;
}
else
{
lean_object* v_reuseFailAlloc_703_; 
v_reuseFailAlloc_703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_703_, 0, v_a_697_);
v___x_702_ = v_reuseFailAlloc_703_;
goto v_reusejp_701_;
}
v_reusejp_701_:
{
return v___x_702_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___redArg___boxed(lean_object* v_a_706_, lean_object* v_b_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___redArg(v_a_706_, v_b_707_, v___y_708_, v___y_709_, v___y_710_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec_ref(v___y_708_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3___lam__0(lean_object* v_numFields_713_, lean_object* v_numParams_714_, lean_object* v_v_715_, lean_object* v_args_716_, lean_object* v_x_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_){
_start:
{
lean_object* v___x_723_; uint8_t v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; 
v___x_723_ = lean_mk_empty_array_with_capacity(v_numFields_713_);
v___x_724_ = 0;
v___x_725_ = lean_array_get_size(v_args_716_);
v___x_726_ = l_Array_toSubarray___redArg(v_args_716_, v_numParams_714_, v___x_725_);
v___x_727_ = lean_box(v___x_724_);
v___x_728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_728_, 0, v___x_723_);
lean_ctor_set(v___x_728_, 1, v___x_727_);
v___x_729_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___redArg(v___x_726_, v___x_728_, v___y_718_, v___y_720_, v___y_721_);
if (lean_obj_tag(v___x_729_) == 0)
{
lean_object* v_a_730_; lean_object* v___x_732_; uint8_t v_isShared_733_; uint8_t v_isSharedCheck_741_; 
v_a_730_ = lean_ctor_get(v___x_729_, 0);
v_isSharedCheck_741_ = !lean_is_exclusive(v___x_729_);
if (v_isSharedCheck_741_ == 0)
{
v___x_732_ = v___x_729_;
v_isShared_733_ = v_isSharedCheck_741_;
goto v_resetjp_731_;
}
else
{
lean_inc(v_a_730_);
lean_dec(v___x_729_);
v___x_732_ = lean_box(0);
v_isShared_733_ = v_isSharedCheck_741_;
goto v_resetjp_731_;
}
v_resetjp_731_:
{
lean_object* v_fst_734_; lean_object* v_snd_735_; lean_object* v___x_736_; uint8_t v___x_737_; lean_object* v___x_739_; 
v_fst_734_ = lean_ctor_get(v_a_730_, 0);
lean_inc(v_fst_734_);
v_snd_735_ = lean_ctor_get(v_a_730_, 1);
lean_inc(v_snd_735_);
lean_dec(v_a_730_);
v___x_736_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_736_, 0, v_v_715_);
lean_ctor_set(v___x_736_, 1, v_fst_734_);
v___x_737_ = lean_unbox(v_snd_735_);
lean_dec(v_snd_735_);
lean_ctor_set_uint8(v___x_736_, sizeof(void*)*2, v___x_737_);
if (v_isShared_733_ == 0)
{
lean_ctor_set(v___x_732_, 0, v___x_736_);
v___x_739_ = v___x_732_;
goto v_reusejp_738_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v___x_736_);
v___x_739_ = v_reuseFailAlloc_740_;
goto v_reusejp_738_;
}
v_reusejp_738_:
{
return v___x_739_;
}
}
}
else
{
lean_object* v_a_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_749_; 
lean_dec(v_v_715_);
v_a_742_ = lean_ctor_get(v___x_729_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_729_);
if (v_isSharedCheck_749_ == 0)
{
v___x_744_ = v___x_729_;
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_a_742_);
lean_dec(v___x_729_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_747_; 
if (v_isShared_745_ == 0)
{
v___x_747_ = v___x_744_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_a_742_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3___lam__0___boxed(lean_object* v_numFields_750_, lean_object* v_numParams_751_, lean_object* v_v_752_, lean_object* v_args_753_, lean_object* v_x_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_, lean_object* v___y_759_){
_start:
{
lean_object* v_res_760_; 
v_res_760_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3___lam__0(v_numFields_750_, v_numParams_751_, v_v_752_, v_args_753_, v_x_754_, v___y_755_, v___y_756_, v___y_757_, v___y_758_);
lean_dec(v___y_758_);
lean_dec_ref(v___y_757_);
lean_dec(v___y_756_);
lean_dec_ref(v___y_755_);
lean_dec_ref(v_x_754_);
lean_dec(v_numFields_750_);
return v_res_760_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3(size_t v_sz_761_, size_t v_i_762_, lean_object* v_bs_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_){
_start:
{
uint8_t v___x_769_; 
v___x_769_ = lean_usize_dec_lt(v_i_762_, v_sz_761_);
if (v___x_769_ == 0)
{
lean_object* v___x_770_; 
v___x_770_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_770_, 0, v_bs_763_);
return v___x_770_;
}
else
{
lean_object* v_v_771_; lean_object* v___x_772_; 
v_v_771_ = lean_array_uget_borrowed(v_bs_763_, v_i_762_);
lean_inc(v_v_771_);
v___x_772_ = lp_aesop_Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0(v_v_771_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_772_) == 0)
{
lean_object* v_a_773_; lean_object* v_toConstantVal_774_; lean_object* v_numParams_775_; lean_object* v_numFields_776_; lean_object* v_type_777_; lean_object* v___f_778_; lean_object* v___x_779_; lean_object* v___x_780_; uint8_t v___x_781_; lean_object* v___x_782_; 
v_a_773_ = lean_ctor_get(v___x_772_, 0);
lean_inc(v_a_773_);
lean_dec_ref_known(v___x_772_, 1);
v_toConstantVal_774_ = lean_ctor_get(v_a_773_, 0);
lean_inc_ref(v_toConstantVal_774_);
v_numParams_775_ = lean_ctor_get(v_a_773_, 3);
lean_inc_n(v_numParams_775_, 2);
v_numFields_776_ = lean_ctor_get(v_a_773_, 4);
lean_inc_n(v_numFields_776_, 2);
lean_dec(v_a_773_);
v_type_777_ = lean_ctor_get(v_toConstantVal_774_, 2);
lean_inc_ref(v_type_777_);
lean_dec_ref(v_toConstantVal_774_);
lean_inc(v_v_771_);
v___f_778_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3___lam__0___boxed), 10, 3);
lean_closure_set(v___f_778_, 0, v_numFields_776_);
lean_closure_set(v___f_778_, 1, v_numParams_775_);
lean_closure_set(v___f_778_, 2, v_v_771_);
v___x_779_ = lean_nat_add(v_numParams_775_, v_numFields_776_);
lean_dec(v_numFields_776_);
lean_dec(v_numParams_775_);
v___x_780_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_780_, 0, v___x_779_);
v___x_781_ = 0;
v___x_782_ = lp_aesop_Lean_Meta_forallBoundedTelescope___at___00Aesop_mkCtorNames_spec__2___redArg(v_type_777_, v___x_780_, v___f_778_, v___x_781_, v___x_781_, v___y_764_, v___y_765_, v___y_766_, v___y_767_);
if (lean_obj_tag(v___x_782_) == 0)
{
lean_object* v_a_783_; lean_object* v___x_784_; lean_object* v_bs_x27_785_; size_t v___x_786_; size_t v___x_787_; lean_object* v___x_788_; 
v_a_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc(v_a_783_);
lean_dec_ref_known(v___x_782_, 1);
v___x_784_ = lean_unsigned_to_nat(0u);
v_bs_x27_785_ = lean_array_uset(v_bs_763_, v_i_762_, v___x_784_);
v___x_786_ = ((size_t)1ULL);
v___x_787_ = lean_usize_add(v_i_762_, v___x_786_);
v___x_788_ = lean_array_uset(v_bs_x27_785_, v_i_762_, v_a_783_);
v_i_762_ = v___x_787_;
v_bs_763_ = v___x_788_;
goto _start;
}
else
{
lean_object* v_a_790_; lean_object* v___x_792_; uint8_t v_isShared_793_; uint8_t v_isSharedCheck_797_; 
lean_dec_ref(v_bs_763_);
v_a_790_ = lean_ctor_get(v___x_782_, 0);
v_isSharedCheck_797_ = !lean_is_exclusive(v___x_782_);
if (v_isSharedCheck_797_ == 0)
{
v___x_792_ = v___x_782_;
v_isShared_793_ = v_isSharedCheck_797_;
goto v_resetjp_791_;
}
else
{
lean_inc(v_a_790_);
lean_dec(v___x_782_);
v___x_792_ = lean_box(0);
v_isShared_793_ = v_isSharedCheck_797_;
goto v_resetjp_791_;
}
v_resetjp_791_:
{
lean_object* v___x_795_; 
if (v_isShared_793_ == 0)
{
v___x_795_ = v___x_792_;
goto v_reusejp_794_;
}
else
{
lean_object* v_reuseFailAlloc_796_; 
v_reuseFailAlloc_796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_796_, 0, v_a_790_);
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
lean_object* v_a_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_805_; 
lean_dec_ref(v_bs_763_);
v_a_798_ = lean_ctor_get(v___x_772_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_772_);
if (v_isSharedCheck_805_ == 0)
{
v___x_800_ = v___x_772_;
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_a_798_);
lean_dec(v___x_772_);
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
v_reuseFailAlloc_804_ = lean_alloc_ctor(1, 1, 0);
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
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3___boxed(lean_object* v_sz_806_, lean_object* v_i_807_, lean_object* v_bs_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_, lean_object* v___y_813_){
_start:
{
size_t v_sz_boxed_814_; size_t v_i_boxed_815_; lean_object* v_res_816_; 
v_sz_boxed_814_ = lean_unbox_usize(v_sz_806_);
lean_dec(v_sz_806_);
v_i_boxed_815_ = lean_unbox_usize(v_i_807_);
lean_dec(v_i_807_);
v_res_816_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3(v_sz_boxed_814_, v_i_boxed_815_, v_bs_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_);
lean_dec(v___y_812_);
lean_dec_ref(v___y_811_);
lean_dec(v___y_810_);
lean_dec_ref(v___y_809_);
return v_res_816_;
}
}
static uint64_t _init_lp_aesop_Aesop_mkCtorNames___closed__1(void){
_start:
{
lean_object* v___x_823_; uint64_t v___x_824_; 
v___x_823_ = ((lean_object*)(lp_aesop_Aesop_mkCtorNames___closed__0));
v___x_824_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_823_);
return v___x_824_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__2(void){
_start:
{
uint64_t v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; 
v___x_825_ = lean_uint64_once(&lp_aesop_Aesop_mkCtorNames___closed__1, &lp_aesop_Aesop_mkCtorNames___closed__1_once, _init_lp_aesop_Aesop_mkCtorNames___closed__1);
v___x_826_ = ((lean_object*)(lp_aesop_Aesop_mkCtorNames___closed__0));
v___x_827_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_827_, 0, v___x_826_);
lean_ctor_set_uint64(v___x_827_, sizeof(void*)*1, v___x_825_);
return v___x_827_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__3(void){
_start:
{
lean_object* v___x_828_; 
v___x_828_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_828_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__4(void){
_start:
{
lean_object* v___x_829_; lean_object* v___x_830_; 
v___x_829_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__3, &lp_aesop_Aesop_mkCtorNames___closed__3_once, _init_lp_aesop_Aesop_mkCtorNames___closed__3);
v___x_830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_830_, 0, v___x_829_);
return v___x_830_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__5(void){
_start:
{
lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; 
v___x_831_ = lean_unsigned_to_nat(32u);
v___x_832_ = lean_mk_empty_array_with_capacity(v___x_831_);
v___x_833_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_833_, 0, v___x_832_);
return v___x_833_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__6(void){
_start:
{
size_t v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; 
v___x_834_ = ((size_t)5ULL);
v___x_835_ = lean_unsigned_to_nat(0u);
v___x_836_ = lean_unsigned_to_nat(32u);
v___x_837_ = lean_mk_empty_array_with_capacity(v___x_836_);
v___x_838_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__5, &lp_aesop_Aesop_mkCtorNames___closed__5_once, _init_lp_aesop_Aesop_mkCtorNames___closed__5);
v___x_839_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_839_, 0, v___x_838_);
lean_ctor_set(v___x_839_, 1, v___x_837_);
lean_ctor_set(v___x_839_, 2, v___x_835_);
lean_ctor_set(v___x_839_, 3, v___x_835_);
lean_ctor_set_usize(v___x_839_, 4, v___x_834_);
return v___x_839_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__7(void){
_start:
{
lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; 
v___x_840_ = lean_box(1);
v___x_841_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__6, &lp_aesop_Aesop_mkCtorNames___closed__6_once, _init_lp_aesop_Aesop_mkCtorNames___closed__6);
v___x_842_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__4, &lp_aesop_Aesop_mkCtorNames___closed__4_once, _init_lp_aesop_Aesop_mkCtorNames___closed__4);
v___x_843_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_843_, 0, v___x_842_);
lean_ctor_set(v___x_843_, 1, v___x_841_);
lean_ctor_set(v___x_843_, 2, v___x_840_);
return v___x_843_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__9(void){
_start:
{
uint8_t v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; uint8_t v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; 
v___x_846_ = 1;
v___x_847_ = lean_unsigned_to_nat(0u);
v___x_848_ = lean_box(0);
v___x_849_ = ((lean_object*)(lp_aesop_Aesop_mkCtorNames___closed__8));
v___x_850_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__7, &lp_aesop_Aesop_mkCtorNames___closed__7_once, _init_lp_aesop_Aesop_mkCtorNames___closed__7);
v___x_851_ = lean_box(1);
v___x_852_ = 0;
v___x_853_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__2, &lp_aesop_Aesop_mkCtorNames___closed__2_once, _init_lp_aesop_Aesop_mkCtorNames___closed__2);
v___x_854_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_854_, 0, v___x_853_);
lean_ctor_set(v___x_854_, 1, v___x_851_);
lean_ctor_set(v___x_854_, 2, v___x_850_);
lean_ctor_set(v___x_854_, 3, v___x_849_);
lean_ctor_set(v___x_854_, 4, v___x_848_);
lean_ctor_set(v___x_854_, 5, v___x_847_);
lean_ctor_set(v___x_854_, 6, v___x_848_);
lean_ctor_set_uint8(v___x_854_, sizeof(void*)*7, v___x_852_);
lean_ctor_set_uint8(v___x_854_, sizeof(void*)*7 + 1, v___x_852_);
lean_ctor_set_uint8(v___x_854_, sizeof(void*)*7 + 2, v___x_852_);
lean_ctor_set_uint8(v___x_854_, sizeof(void*)*7 + 3, v___x_846_);
return v___x_854_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__10(void){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v___x_855_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__4, &lp_aesop_Aesop_mkCtorNames___closed__4_once, _init_lp_aesop_Aesop_mkCtorNames___closed__4);
v___x_856_ = lean_unsigned_to_nat(0u);
v___x_857_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_857_, 0, v___x_856_);
lean_ctor_set(v___x_857_, 1, v___x_856_);
lean_ctor_set(v___x_857_, 2, v___x_856_);
lean_ctor_set(v___x_857_, 3, v___x_856_);
lean_ctor_set(v___x_857_, 4, v___x_855_);
lean_ctor_set(v___x_857_, 5, v___x_855_);
lean_ctor_set(v___x_857_, 6, v___x_855_);
lean_ctor_set(v___x_857_, 7, v___x_855_);
lean_ctor_set(v___x_857_, 8, v___x_855_);
lean_ctor_set(v___x_857_, 9, v___x_855_);
return v___x_857_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__11(void){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; 
v___x_858_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__4, &lp_aesop_Aesop_mkCtorNames___closed__4_once, _init_lp_aesop_Aesop_mkCtorNames___closed__4);
v___x_859_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_859_, 0, v___x_858_);
lean_ctor_set(v___x_859_, 1, v___x_858_);
lean_ctor_set(v___x_859_, 2, v___x_858_);
lean_ctor_set(v___x_859_, 3, v___x_858_);
lean_ctor_set(v___x_859_, 4, v___x_858_);
lean_ctor_set(v___x_859_, 5, v___x_858_);
return v___x_859_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__12(void){
_start:
{
lean_object* v___x_860_; lean_object* v___x_861_; 
v___x_860_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__4, &lp_aesop_Aesop_mkCtorNames___closed__4_once, _init_lp_aesop_Aesop_mkCtorNames___closed__4);
v___x_861_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_861_, 0, v___x_860_);
lean_ctor_set(v___x_861_, 1, v___x_860_);
lean_ctor_set(v___x_861_, 2, v___x_860_);
lean_ctor_set(v___x_861_, 3, v___x_860_);
lean_ctor_set(v___x_861_, 4, v___x_860_);
return v___x_861_;
}
}
static lean_object* _init_lp_aesop_Aesop_mkCtorNames___closed__13(void){
_start:
{
lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; 
v___x_862_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__12, &lp_aesop_Aesop_mkCtorNames___closed__12_once, _init_lp_aesop_Aesop_mkCtorNames___closed__12);
v___x_863_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__6, &lp_aesop_Aesop_mkCtorNames___closed__6_once, _init_lp_aesop_Aesop_mkCtorNames___closed__6);
v___x_864_ = lean_box(1);
v___x_865_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__11, &lp_aesop_Aesop_mkCtorNames___closed__11_once, _init_lp_aesop_Aesop_mkCtorNames___closed__11);
v___x_866_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__10, &lp_aesop_Aesop_mkCtorNames___closed__10_once, _init_lp_aesop_Aesop_mkCtorNames___closed__10);
v___x_867_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_867_, 0, v___x_866_);
lean_ctor_set(v___x_867_, 1, v___x_865_);
lean_ctor_set(v___x_867_, 2, v___x_864_);
lean_ctor_set(v___x_867_, 3, v___x_863_);
lean_ctor_set(v___x_867_, 4, v___x_862_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkCtorNames(lean_object* v_iv_868_, lean_object* v_a_869_, lean_object* v_a_870_){
_start:
{
lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v_ctors_875_; lean_object* v___x_876_; size_t v_sz_877_; size_t v___x_878_; lean_object* v___x_879_; 
v___x_872_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__9, &lp_aesop_Aesop_mkCtorNames___closed__9_once, _init_lp_aesop_Aesop_mkCtorNames___closed__9);
v___x_873_ = lean_obj_once(&lp_aesop_Aesop_mkCtorNames___closed__13, &lp_aesop_Aesop_mkCtorNames___closed__13_once, _init_lp_aesop_Aesop_mkCtorNames___closed__13);
v___x_874_ = lean_st_mk_ref(v___x_873_);
v_ctors_875_ = lean_ctor_get(v_iv_868_, 4);
lean_inc(v_ctors_875_);
lean_dec_ref(v_iv_868_);
v___x_876_ = lean_array_mk(v_ctors_875_);
v_sz_877_ = lean_array_size(v___x_876_);
v___x_878_ = ((size_t)0ULL);
v___x_879_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_mkCtorNames_spec__3(v_sz_877_, v___x_878_, v___x_876_, v___x_872_, v___x_874_, v_a_869_, v_a_870_);
if (lean_obj_tag(v___x_879_) == 0)
{
lean_object* v_a_880_; lean_object* v___x_882_; uint8_t v_isShared_883_; uint8_t v_isSharedCheck_888_; 
v_a_880_ = lean_ctor_get(v___x_879_, 0);
v_isSharedCheck_888_ = !lean_is_exclusive(v___x_879_);
if (v_isSharedCheck_888_ == 0)
{
v___x_882_ = v___x_879_;
v_isShared_883_ = v_isSharedCheck_888_;
goto v_resetjp_881_;
}
else
{
lean_inc(v_a_880_);
lean_dec(v___x_879_);
v___x_882_ = lean_box(0);
v_isShared_883_ = v_isSharedCheck_888_;
goto v_resetjp_881_;
}
v_resetjp_881_:
{
lean_object* v___x_884_; lean_object* v___x_886_; 
v___x_884_ = lean_st_ref_get(v___x_874_);
lean_dec(v___x_874_);
lean_dec(v___x_884_);
if (v_isShared_883_ == 0)
{
v___x_886_ = v___x_882_;
goto v_reusejp_885_;
}
else
{
lean_object* v_reuseFailAlloc_887_; 
v_reuseFailAlloc_887_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_887_, 0, v_a_880_);
v___x_886_ = v_reuseFailAlloc_887_;
goto v_reusejp_885_;
}
v_reusejp_885_:
{
return v___x_886_;
}
}
}
else
{
lean_dec(v___x_874_);
return v___x_879_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkCtorNames___boxed(lean_object* v_iv_889_, lean_object* v_a_890_, lean_object* v_a_891_, lean_object* v_a_892_){
_start:
{
lean_object* v_res_893_; 
v_res_893_ = lp_aesop_Aesop_mkCtorNames(v_iv_889_, v_a_890_, v_a_891_);
lean_dec(v_a_891_);
lean_dec_ref(v_a_890_);
return v_res_893_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1(lean_object* v_inst_894_, lean_object* v_R_895_, lean_object* v_a_896_, lean_object* v_b_897_, lean_object* v_c_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_){
_start:
{
lean_object* v___x_904_; 
v___x_904_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___redArg(v_a_896_, v_b_897_, v___y_899_, v___y_901_, v___y_902_);
return v___x_904_;
}
}
LEAN_EXPORT lean_object* lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1___boxed(lean_object* v_inst_905_, lean_object* v_R_906_, lean_object* v_a_907_, lean_object* v_b_908_, lean_object* v_c_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_aesop_WellFounded_opaqueFix_u2083___at___00Aesop_mkCtorNames_spec__1(v_inst_905_, v_R_906_, v_a_907_, v_b_908_, v_c_909_, v___y_910_, v___y_911_, v___y_912_, v___y_913_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
lean_dec(v___y_911_);
lean_dec_ref(v___y_910_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0(lean_object* v_00_u03b1_916_, lean_object* v_msg_917_, lean_object* v___y_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_){
_start:
{
lean_object* v___x_923_; 
v___x_923_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___redArg(v_msg_917_, v___y_918_, v___y_919_, v___y_920_, v___y_921_);
return v___x_923_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0___boxed(lean_object* v_00_u03b1_924_, lean_object* v_msg_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_aesop_Lean_throwError___at___00Lean_getConstInfoCtor___at___00Aesop_mkCtorNames_spec__0_spec__0(v_00_u03b1_924_, v_msg_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_);
lean_dec(v___y_929_);
lean_dec_ref(v___y_928_);
lean_dec(v___y_927_);
lean_dec_ref(v___y_926_);
return v_res_931_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Induction(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_CtorNames(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_CtorNames(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Tactic_Induction(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_CtorNames(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_CtorNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_CtorNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_CtorNames(builtin);
}
#ifdef __cplusplus
}
#endif
