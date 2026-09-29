// Lean compiler output
// Module: Mathlib.Tactic.Polynomial.Core
// Imports: public import Init public meta import Init meta import Lean.Compiler.IR.CompilerM public meta import Lean.Meta.Tactic.Simp.Attr public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_io_user_error(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Environment_evalConstCheck___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerPersistentEnvExtensionUnsafe___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_registerSimpAttr(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_decl_get_sorry_dep(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_instBEqAttributeKind_beq(uint8_t, uint8_t);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "polynomial_pre"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 143, 218, 102, 255, 139, 96, 24)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 111, .m_capacity = 111, .m_length = 110, .m_data = "The `polynomial_pre` simp attribute uses preprocessing lemmas to turn specialized functions into `algebraMap`s"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Polynomial"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "polynomialPreExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(65, 200, 72, 213, 209, 94, 127, 253)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(88, 8, 157, 119, 151, 7, 250, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_polynomialPreExt;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "polynomial_post"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(162, 81, 229, 25, 250, 253, 84, 253)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 119, .m_capacity = 119, .m_length = 118, .m_data = "The `polynomial_post` simp attribute uses postprocessing lemmas to turn `algebraMap`s into more specialized functions."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "polynomialPostExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(65, 200, 72, 213, 209, 94, 127, 253)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(5, 195, 62, 182, 71, 234, 124, 54)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_polynomialPostExt;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "PolyInferBaseAttr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(65, 200, 72, 213, 209, 94, 127, 253)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(96, 37, 57, 225, 146, 28, 130, 215)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "polynomial_infer_base"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "PolynomialExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(65, 200, 72, 213, 209, 94, 127, 253)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(207, 150, 38, 62, 153, 101, 4, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_mkPolynomialExt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_mkPolynomialExt___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__3_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__3_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "polynomialExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(65, 200, 72, 213, 209, 94, 127, 253)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(128, 210, 220, 178, 110, 7, 16, 137)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__8_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__8_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__8_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__9_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 0, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__8_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__9_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__9_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__10_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__9_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__10_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__10_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_polynomialExt;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 79, .m_data = "invalid attribute 'polynomial_infer_base', declaration is in an imported module"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "invalid attribute 'polynomial_infer_base', must be global"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(253, 45, 17, 246, 249, 145, 51, 82)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Core"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(21, 251, 13, 212, 90, 15, 4, 79)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__6_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(128, 68, 109, 162, 254, 125, 67, 204)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__8_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(137, 78, 16, 220, 144, 137, 162, 127)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__8_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__8_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__9_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__8_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(224, 43, 18, 123, 105, 58, 36, 130)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__9_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__9_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__10_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__9_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(78, 134, 183, 119, 248, 16, 155, 27)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__10_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__10_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__11_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__11_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__11_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__12_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__10_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__11_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(99, 240, 52, 146, 112, 249, 48, 86)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__12_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__12_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__13_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__13_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__13_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__14_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__12_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__13_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(86, 164, 13, 60, 128, 21, 72, 156)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__14_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__14_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__15_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__14_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(111, 36, 50, 79, 171, 209, 65, 225)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__15_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__15_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__16_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__15_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(254, 9, 158, 226, 52, 23, 104, 8)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__16_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__16_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__17_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__16_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(40, 138, 22, 18, 4, 200, 101, 65)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__17_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__17_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__18_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__17_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(228, 96, 192, 29, 162, 171, 138, 221)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__18_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__18_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__19_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__18_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)(((size_t)(737497776) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(158, 137, 114, 193, 69, 212, 228, 226)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__19_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__19_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__20_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__20_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__20_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__21_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__19_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__20_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(137, 53, 18, 52, 22, 134, 189, 188)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__21_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__21_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__22_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__22_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__22_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__23_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__21_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__22_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(161, 105, 162, 58, 17, 97, 92, 28)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__23_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__23_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__24_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__23_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(124, 58, 12, 83, 219, 121, 99, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__24_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__24_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__25_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2____boxed, .m_arity = 11, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__5_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__25_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__25_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__26_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_PolyInferBaseAttr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 230, 233, 41, 115, 9, 223, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__26_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__26_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__27_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__26_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__27_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__27_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__28_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 79, .m_data = "adds a polynomial extension that infers the base ring of a polynomial-like type"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__28_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__28_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__29_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__24_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__26_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__28_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__29_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__29_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__30_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__29_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__25_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__27_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__30_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__30_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_inferBase(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_15_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_));
v___x_16_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_));
v___x_17_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__7_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_));
v___x_18_ = l_Lean_Meta_registerSimpAttr(v___x_15_, v___x_16_, v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2____boxed(lean_object* v_a_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_();
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_32_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_));
v___x_33_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_));
v___x_34_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__4_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_));
v___x_35_ = l_Lean_Meta_registerSimpAttr(v___x_32_, v___x_33_, v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2____boxed(lean_object* v_a_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_();
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1(lean_object* v_n_59_, lean_object* v_env_60_, lean_object* v_opts_61_){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1));
v___x_63_ = l_Lean_Environment_evalConstCheck___redArg(v_env_60_, v_opts_61_, v___x_62_, v_n_59_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___boxed(lean_object* v_n_64_, lean_object* v_env_65_, lean_object* v_opts_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1(v_n_64_, v_env_65_, v_opts_66_);
lean_dec_ref(v_opts_66_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___redArg(lean_object* v_e_68_){
_start:
{
if (lean_obj_tag(v_e_68_) == 0)
{
lean_object* v_a_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_78_; 
v_a_70_ = lean_ctor_get(v_e_68_, 0);
v_isSharedCheck_78_ = !lean_is_exclusive(v_e_68_);
if (v_isSharedCheck_78_ == 0)
{
v___x_72_ = v_e_68_;
v_isShared_73_ = v_isSharedCheck_78_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_a_70_);
lean_dec(v_e_68_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_78_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___x_74_; lean_object* v___x_76_; 
v___x_74_ = lean_mk_io_user_error(v_a_70_);
if (v_isShared_73_ == 0)
{
lean_ctor_set_tag(v___x_72_, 1);
lean_ctor_set(v___x_72_, 0, v___x_74_);
v___x_76_ = v___x_72_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v___x_74_);
v___x_76_ = v_reuseFailAlloc_77_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
return v___x_76_;
}
}
}
else
{
lean_object* v_a_79_; lean_object* v___x_81_; uint8_t v_isShared_82_; uint8_t v_isSharedCheck_86_; 
v_a_79_ = lean_ctor_get(v_e_68_, 0);
v_isSharedCheck_86_ = !lean_is_exclusive(v_e_68_);
if (v_isSharedCheck_86_ == 0)
{
v___x_81_ = v_e_68_;
v_isShared_82_ = v_isSharedCheck_86_;
goto v_resetjp_80_;
}
else
{
lean_inc(v_a_79_);
lean_dec(v_e_68_);
v___x_81_ = lean_box(0);
v_isShared_82_ = v_isSharedCheck_86_;
goto v_resetjp_80_;
}
v_resetjp_80_:
{
lean_object* v___x_84_; 
if (v_isShared_82_ == 0)
{
lean_ctor_set_tag(v___x_81_, 0);
v___x_84_ = v___x_81_;
goto v_reusejp_83_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v_a_79_);
v___x_84_ = v_reuseFailAlloc_85_;
goto v_reusejp_83_;
}
v_reusejp_83_:
{
return v___x_84_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___redArg___boxed(lean_object* v_e_87_, lean_object* v_a_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___redArg(v_e_87_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0(lean_object* v_00_u03b1_90_, lean_object* v_e_91_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___redArg(v_e_91_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___boxed(lean_object* v_00_u03b1_94_, lean_object* v_e_95_, lean_object* v_a_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0(v_00_u03b1_94_, v_e_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_mkPolynomialExt(lean_object* v_n_98_, lean_object* v_a_99_){
_start:
{
lean_object* v_env_101_; lean_object* v_opts_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v_env_101_ = lean_ctor_get(v_a_99_, 0);
v_opts_102_ = lean_ctor_get(v_a_99_, 1);
v___x_103_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_mkPolynomialExt_unsafe__1___closed__1));
lean_inc_ref(v_env_101_);
v___x_104_ = l_Lean_Environment_evalConstCheck___redArg(v_env_101_, v_opts_102_, v___x_103_, v_n_98_);
v___x_105_ = lp_mathlib_IO_ofExcept___at___00Mathlib_Tactic_Polynomial_mkPolynomialExt_spec__0___redArg(v___x_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_mkPolynomialExt___boxed(lean_object* v_n_106_, lean_object* v_a_107_, lean_object* v_a_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_Mathlib_Tactic_Polynomial_mkPolynomialExt(v_n_106_, v_a_107_);
lean_dec_ref(v_a_107_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object* v_x_110_, lean_object* v_x_111_){
_start:
{
lean_object* v_fst_112_; lean_object* v_snd_113_; lean_object* v___x_115_; uint8_t v_isShared_116_; uint8_t v_isSharedCheck_123_; 
v_fst_112_ = lean_ctor_get(v_x_110_, 0);
v_snd_113_ = lean_ctor_get(v_x_110_, 1);
v_isSharedCheck_123_ = !lean_is_exclusive(v_x_110_);
if (v_isSharedCheck_123_ == 0)
{
v___x_115_ = v_x_110_;
v_isShared_116_ = v_isSharedCheck_123_;
goto v_resetjp_114_;
}
else
{
lean_inc(v_snd_113_);
lean_inc(v_fst_112_);
lean_dec(v_x_110_);
v___x_115_ = lean_box(0);
v_isShared_116_ = v_isSharedCheck_123_;
goto v_resetjp_114_;
}
v_resetjp_114_:
{
lean_object* v_fst_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_121_; 
v_fst_117_ = lean_ctor_get(v_x_111_, 0);
lean_inc(v_fst_117_);
v___x_118_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_118_, 0, v_fst_117_);
lean_ctor_set(v___x_118_, 1, v_fst_112_);
v___x_119_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_119_, 0, v_x_111_);
lean_ctor_set(v___x_119_, 1, v_snd_113_);
if (v_isShared_116_ == 0)
{
lean_ctor_set(v___x_115_, 1, v___x_119_);
lean_ctor_set(v___x_115_, 0, v___x_118_);
v___x_121_ = v___x_115_;
goto v_reusejp_120_;
}
else
{
lean_object* v_reuseFailAlloc_122_; 
v_reuseFailAlloc_122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_122_, 0, v___x_118_);
lean_ctor_set(v_reuseFailAlloc_122_, 1, v___x_119_);
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
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object* v_x_124_, lean_object* v_s_125_){
_start:
{
lean_object* v_fst_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v_fst_126_ = lean_ctor_get(v_s_125_, 0);
lean_inc(v_fst_126_);
lean_dec_ref(v_s_125_);
v___x_127_ = l_List_reverse___redArg(v_fst_126_);
v___x_128_ = lean_array_mk(v___x_127_);
lean_inc_ref_n(v___x_128_, 2);
v___x_129_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_129_, 0, v___x_128_);
lean_ctor_set(v___x_129_, 1, v___x_128_);
lean_ctor_set(v___x_129_, 2, v___x_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object* v_x_130_, lean_object* v_s_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(v_x_130_, v_s_131_);
lean_dec_ref(v_x_130_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object* v_x_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lean_box(0);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object* v_x_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__2_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(v_x_135_);
lean_dec_ref(v_x_135_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__3_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object* v_s_137_){
_start:
{
lean_object* v_fst_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v_fst_138_ = lean_ctor_get(v_s_137_, 0);
lean_inc(v_fst_138_);
lean_dec_ref(v_s_137_);
v___x_139_ = l_List_reverse___redArg(v_fst_138_);
v___x_140_ = lean_array_mk(v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__0(lean_object* v_as_141_, size_t v_i_142_, size_t v_stop_143_, lean_object* v_b_144_, lean_object* v___y_145_){
_start:
{
uint8_t v___x_147_; 
v___x_147_ = lean_usize_dec_eq(v_i_142_, v_stop_143_);
if (v___x_147_ == 0)
{
lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_148_ = lean_array_uget_borrowed(v_as_141_, v_i_142_);
lean_inc(v___x_148_);
v___x_149_ = lp_mathlib_Mathlib_Tactic_Polynomial_mkPolynomialExt(v___x_148_, v___y_145_);
if (lean_obj_tag(v___x_149_) == 0)
{
lean_object* v_a_150_; lean_object* v___x_151_; lean_object* v___x_152_; size_t v___x_153_; size_t v___x_154_; 
v_a_150_ = lean_ctor_get(v___x_149_, 0);
lean_inc(v_a_150_);
lean_dec_ref_known(v___x_149_, 1);
lean_inc(v___x_148_);
v___x_151_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_151_, 0, v___x_148_);
lean_ctor_set(v___x_151_, 1, v_a_150_);
v___x_152_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v_b_144_);
v___x_153_ = ((size_t)1ULL);
v___x_154_ = lean_usize_add(v_i_142_, v___x_153_);
v_i_142_ = v___x_154_;
v_b_144_ = v___x_152_;
goto _start;
}
else
{
lean_object* v_a_156_; lean_object* v___x_158_; uint8_t v_isShared_159_; uint8_t v_isSharedCheck_163_; 
lean_dec(v_b_144_);
v_a_156_ = lean_ctor_get(v___x_149_, 0);
v_isSharedCheck_163_ = !lean_is_exclusive(v___x_149_);
if (v_isSharedCheck_163_ == 0)
{
v___x_158_ = v___x_149_;
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
else
{
lean_inc(v_a_156_);
lean_dec(v___x_149_);
v___x_158_ = lean_box(0);
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
v_resetjp_157_:
{
lean_object* v___x_161_; 
if (v_isShared_159_ == 0)
{
v___x_161_ = v___x_158_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_a_156_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
}
}
else
{
lean_object* v___x_164_; 
v___x_164_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_164_, 0, v_b_144_);
return v___x_164_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__0___boxed(lean_object* v_as_165_, lean_object* v_i_166_, lean_object* v_stop_167_, lean_object* v_b_168_, lean_object* v___y_169_, lean_object* v___y_170_){
_start:
{
size_t v_i_boxed_171_; size_t v_stop_boxed_172_; lean_object* v_res_173_; 
v_i_boxed_171_ = lean_unbox_usize(v_i_166_);
lean_dec(v_i_166_);
v_stop_boxed_172_ = lean_unbox_usize(v_stop_167_);
lean_dec(v_stop_167_);
v_res_173_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__0(v_as_165_, v_i_boxed_171_, v_stop_boxed_172_, v_b_168_, v___y_169_);
lean_dec_ref(v___y_169_);
lean_dec_ref(v_as_165_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__1(lean_object* v_as_174_, size_t v_i_175_, size_t v_stop_176_, lean_object* v_b_177_, lean_object* v___y_178_){
_start:
{
lean_object* v_a_181_; lean_object* v___y_186_; uint8_t v___x_188_; 
v___x_188_ = lean_usize_dec_eq(v_i_175_, v_stop_176_);
if (v___x_188_ == 0)
{
lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; uint8_t v___x_192_; 
v___x_189_ = lean_array_uget_borrowed(v_as_174_, v_i_175_);
v___x_190_ = lean_unsigned_to_nat(0u);
v___x_191_ = lean_array_get_size(v___x_189_);
v___x_192_ = lean_nat_dec_lt(v___x_190_, v___x_191_);
if (v___x_192_ == 0)
{
v_a_181_ = v_b_177_;
goto v___jp_180_;
}
else
{
uint8_t v___x_193_; 
v___x_193_ = lean_nat_dec_le(v___x_191_, v___x_191_);
if (v___x_193_ == 0)
{
if (v___x_192_ == 0)
{
v_a_181_ = v_b_177_;
goto v___jp_180_;
}
else
{
size_t v___x_194_; size_t v___x_195_; lean_object* v___x_196_; 
v___x_194_ = ((size_t)0ULL);
v___x_195_ = lean_usize_of_nat(v___x_191_);
v___x_196_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__0(v___x_189_, v___x_194_, v___x_195_, v_b_177_, v___y_178_);
v___y_186_ = v___x_196_;
goto v___jp_185_;
}
}
else
{
size_t v___x_197_; size_t v___x_198_; lean_object* v___x_199_; 
v___x_197_ = ((size_t)0ULL);
v___x_198_ = lean_usize_of_nat(v___x_191_);
v___x_199_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__0(v___x_189_, v___x_197_, v___x_198_, v_b_177_, v___y_178_);
v___y_186_ = v___x_199_;
goto v___jp_185_;
}
}
}
else
{
lean_object* v___x_200_; 
v___x_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_200_, 0, v_b_177_);
return v___x_200_;
}
v___jp_180_:
{
size_t v___x_182_; size_t v___x_183_; 
v___x_182_ = ((size_t)1ULL);
v___x_183_ = lean_usize_add(v_i_175_, v___x_182_);
v_i_175_ = v___x_183_;
v_b_177_ = v_a_181_;
goto _start;
}
v___jp_185_:
{
if (lean_obj_tag(v___y_186_) == 0)
{
lean_object* v_a_187_; 
v_a_187_ = lean_ctor_get(v___y_186_, 0);
lean_inc(v_a_187_);
lean_dec_ref_known(v___y_186_, 1);
v_a_181_ = v_a_187_;
goto v___jp_180_;
}
else
{
return v___y_186_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__1___boxed(lean_object* v_as_201_, lean_object* v_i_202_, lean_object* v_stop_203_, lean_object* v_b_204_, lean_object* v___y_205_, lean_object* v___y_206_){
_start:
{
size_t v_i_boxed_207_; size_t v_stop_boxed_208_; lean_object* v_res_209_; 
v_i_boxed_207_ = lean_unbox_usize(v_i_202_);
lean_dec(v_i_202_);
v_stop_boxed_208_ = lean_unbox_usize(v_stop_203_);
lean_dec(v_stop_203_);
v_res_209_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__1(v_as_201_, v_i_boxed_207_, v_stop_boxed_208_, v_b_204_, v___y_205_);
lean_dec_ref(v___y_205_);
lean_dec_ref(v_as_201_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object* v___x_210_, lean_object* v___x_211_, lean_object* v_s_212_, lean_object* v___y_213_){
_start:
{
lean_object* v_a_216_; lean_object* v___y_220_; lean_object* v___x_230_; lean_object* v___x_231_; uint8_t v___x_232_; 
v___x_230_ = lean_unsigned_to_nat(0u);
v___x_231_ = lean_array_get_size(v_s_212_);
v___x_232_ = lean_nat_dec_lt(v___x_230_, v___x_231_);
if (v___x_232_ == 0)
{
v_a_216_ = v___x_211_;
goto v___jp_215_;
}
else
{
uint8_t v___x_233_; 
v___x_233_ = lean_nat_dec_le(v___x_231_, v___x_231_);
if (v___x_233_ == 0)
{
if (v___x_232_ == 0)
{
v_a_216_ = v___x_211_;
goto v___jp_215_;
}
else
{
size_t v___x_234_; size_t v___x_235_; lean_object* v___x_236_; 
v___x_234_ = ((size_t)0ULL);
v___x_235_ = lean_usize_of_nat(v___x_231_);
v___x_236_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__1(v_s_212_, v___x_234_, v___x_235_, v___x_211_, v___y_213_);
v___y_220_ = v___x_236_;
goto v___jp_219_;
}
}
else
{
size_t v___x_237_; size_t v___x_238_; lean_object* v___x_239_; 
v___x_237_ = ((size_t)0ULL);
v___x_238_ = lean_usize_of_nat(v___x_231_);
v___x_239_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2__spec__1(v_s_212_, v___x_237_, v___x_238_, v___x_211_, v___y_213_);
v___y_220_ = v___x_239_;
goto v___jp_219_;
}
}
v___jp_215_:
{
lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_217_, 0, v___x_210_);
lean_ctor_set(v___x_217_, 1, v_a_216_);
v___x_218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
return v___x_218_;
}
v___jp_219_:
{
if (lean_obj_tag(v___y_220_) == 0)
{
lean_object* v_a_221_; 
v_a_221_ = lean_ctor_get(v___y_220_, 0);
lean_inc(v_a_221_);
lean_dec_ref_known(v___y_220_, 1);
v_a_216_ = v_a_221_;
goto v___jp_215_;
}
else
{
lean_object* v_a_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_229_; 
lean_dec(v___x_210_);
v_a_222_ = lean_ctor_get(v___y_220_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v___y_220_);
if (v_isSharedCheck_229_ == 0)
{
v___x_224_ = v___y_220_;
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_a_222_);
lean_dec(v___y_220_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v___x_227_; 
if (v_isShared_225_ == 0)
{
v___x_227_ = v___x_224_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v_a_222_);
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
return v___x_227_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object* v___x_240_, lean_object* v___x_241_, lean_object* v_s_242_, lean_object* v___y_243_, lean_object* v___y_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__4_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(v___x_240_, v___x_241_, v_s_242_, v___y_243_);
lean_dec_ref(v___y_243_);
lean_dec_ref(v_s_242_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(lean_object* v___x_246_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_248_, 0, v___x_246_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object* v___x_249_, lean_object* v___y_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__5_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(v___x_249_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_281_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__10_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_));
v___x_282_ = l_Lean_registerPersistentEnvExtensionUnsafe___redArg(v___x_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2____boxed(lean_object* v_a_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_();
return v_res_284_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_285_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_286_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_287_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_287_, 0, v___x_286_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_288_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__1, &lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__1);
v___x_289_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_288_);
lean_ctor_set(v___x_289_, 1, v___x_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg(lean_object* v_env_290_, lean_object* v___y_291_){
_start:
{
lean_object* v___x_293_; lean_object* v_nextMacroScope_294_; lean_object* v_ngen_295_; lean_object* v_auxDeclNGen_296_; lean_object* v_traceState_297_; lean_object* v_messages_298_; lean_object* v_infoState_299_; lean_object* v_snapshotTasks_300_; lean_object* v___x_302_; uint8_t v_isShared_303_; uint8_t v_isSharedCheck_311_; 
v___x_293_ = lean_st_ref_take(v___y_291_);
v_nextMacroScope_294_ = lean_ctor_get(v___x_293_, 1);
v_ngen_295_ = lean_ctor_get(v___x_293_, 2);
v_auxDeclNGen_296_ = lean_ctor_get(v___x_293_, 3);
v_traceState_297_ = lean_ctor_get(v___x_293_, 4);
v_messages_298_ = lean_ctor_get(v___x_293_, 6);
v_infoState_299_ = lean_ctor_get(v___x_293_, 7);
v_snapshotTasks_300_ = lean_ctor_get(v___x_293_, 8);
v_isSharedCheck_311_ = !lean_is_exclusive(v___x_293_);
if (v_isSharedCheck_311_ == 0)
{
lean_object* v_unused_312_; lean_object* v_unused_313_; 
v_unused_312_ = lean_ctor_get(v___x_293_, 5);
lean_dec(v_unused_312_);
v_unused_313_ = lean_ctor_get(v___x_293_, 0);
lean_dec(v_unused_313_);
v___x_302_ = v___x_293_;
v_isShared_303_ = v_isSharedCheck_311_;
goto v_resetjp_301_;
}
else
{
lean_inc(v_snapshotTasks_300_);
lean_inc(v_infoState_299_);
lean_inc(v_messages_298_);
lean_inc(v_traceState_297_);
lean_inc(v_auxDeclNGen_296_);
lean_inc(v_ngen_295_);
lean_inc(v_nextMacroScope_294_);
lean_dec(v___x_293_);
v___x_302_ = lean_box(0);
v_isShared_303_ = v_isSharedCheck_311_;
goto v_resetjp_301_;
}
v_resetjp_301_:
{
lean_object* v___x_304_; lean_object* v___x_306_; 
v___x_304_ = lean_obj_once(&lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__2, &lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___closed__2);
if (v_isShared_303_ == 0)
{
lean_ctor_set(v___x_302_, 5, v___x_304_);
lean_ctor_set(v___x_302_, 0, v_env_290_);
v___x_306_ = v___x_302_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_310_; 
v_reuseFailAlloc_310_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_310_, 0, v_env_290_);
lean_ctor_set(v_reuseFailAlloc_310_, 1, v_nextMacroScope_294_);
lean_ctor_set(v_reuseFailAlloc_310_, 2, v_ngen_295_);
lean_ctor_set(v_reuseFailAlloc_310_, 3, v_auxDeclNGen_296_);
lean_ctor_set(v_reuseFailAlloc_310_, 4, v_traceState_297_);
lean_ctor_set(v_reuseFailAlloc_310_, 5, v___x_304_);
lean_ctor_set(v_reuseFailAlloc_310_, 6, v_messages_298_);
lean_ctor_set(v_reuseFailAlloc_310_, 7, v_infoState_299_);
lean_ctor_set(v_reuseFailAlloc_310_, 8, v_snapshotTasks_300_);
v___x_306_ = v_reuseFailAlloc_310_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v___x_307_ = lean_st_ref_set(v___y_291_, v___x_306_);
v___x_308_ = lean_box(0);
v___x_309_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
return v___x_309_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v_env_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg(v_env_314_, v___y_315_);
lean_dec(v___y_315_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0(lean_object* v_env_318_, lean_object* v___y_319_, lean_object* v___y_320_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg(v_env_318_, v___y_320_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___boxed(lean_object* v_env_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0(v_env_323_, v___y_324_, v___y_325_);
lean_dec(v___y_325_);
lean_dec_ref(v___y_324_);
return v_res_327_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_box(0);
v___x_329_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_330_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_330_, 0, v___x_329_);
lean_ctor_set(v___x_330_, 1, v___x_328_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg(){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_332_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg___closed__0);
v___x_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_333_, 0, v___x_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object* v___y_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg();
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2(lean_object* v_00_u03b1_336_, lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg();
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___boxed(lean_object* v_00_u03b1_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2(v_00_u03b1_341_, v___y_342_, v___y_343_);
lean_dec(v___y_343_);
lean_dec_ref(v___y_342_);
return v_res_345_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__0(void){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_346_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1(void){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_347_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__0);
v___x_348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
return v___x_348_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__2(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_349_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1);
v___x_350_ = lean_unsigned_to_nat(0u);
v___x_351_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_351_, 0, v___x_350_);
lean_ctor_set(v___x_351_, 1, v___x_350_);
lean_ctor_set(v___x_351_, 2, v___x_350_);
lean_ctor_set(v___x_351_, 3, v___x_350_);
lean_ctor_set(v___x_351_, 4, v___x_349_);
lean_ctor_set(v___x_351_, 5, v___x_349_);
lean_ctor_set(v___x_351_, 6, v___x_349_);
lean_ctor_set(v___x_351_, 7, v___x_349_);
lean_ctor_set(v___x_351_, 8, v___x_349_);
lean_ctor_set(v___x_351_, 9, v___x_349_);
return v___x_351_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__3(void){
_start:
{
lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_352_ = lean_unsigned_to_nat(32u);
v___x_353_ = lean_mk_empty_array_with_capacity(v___x_352_);
v___x_354_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_354_, 0, v___x_353_);
return v___x_354_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__4(void){
_start:
{
size_t v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_355_ = ((size_t)5ULL);
v___x_356_ = lean_unsigned_to_nat(0u);
v___x_357_ = lean_unsigned_to_nat(32u);
v___x_358_ = lean_mk_empty_array_with_capacity(v___x_357_);
v___x_359_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__3);
v___x_360_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_360_, 0, v___x_359_);
lean_ctor_set(v___x_360_, 1, v___x_358_);
lean_ctor_set(v___x_360_, 2, v___x_356_);
lean_ctor_set(v___x_360_, 3, v___x_356_);
lean_ctor_set_usize(v___x_360_, 4, v___x_355_);
return v___x_360_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__5(void){
_start:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_361_ = lean_box(1);
v___x_362_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__4);
v___x_363_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__1);
v___x_364_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v___x_362_);
lean_ctor_set(v___x_364_, 2, v___x_361_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1(lean_object* v_msgData_365_, lean_object* v___y_366_, lean_object* v___y_367_){
_start:
{
lean_object* v___x_369_; lean_object* v_env_370_; lean_object* v_options_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_369_ = lean_st_ref_get(v___y_367_);
v_env_370_ = lean_ctor_get(v___x_369_, 0);
lean_inc_ref(v_env_370_);
lean_dec(v___x_369_);
v_options_371_ = lean_ctor_get(v___y_366_, 2);
v___x_372_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__2);
v___x_373_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___closed__5);
lean_inc_ref(v_options_371_);
v___x_374_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_374_, 0, v_env_370_);
lean_ctor_set(v___x_374_, 1, v___x_372_);
lean_ctor_set(v___x_374_, 2, v___x_373_);
lean_ctor_set(v___x_374_, 3, v_options_371_);
v___x_375_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v_msgData_365_);
v___x_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1___boxed(lean_object* v_msgData_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1(v_msgData_377_, v___y_378_, v___y_379_);
lean_dec(v___y_379_);
lean_dec_ref(v___y_378_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg(lean_object* v_msg_382_, lean_object* v___y_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_ref_386_; lean_object* v___x_387_; lean_object* v_a_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_396_; 
v_ref_386_ = lean_ctor_get(v___y_383_, 5);
v___x_387_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1_spec__1(v_msg_382_, v___y_383_, v___y_384_);
v_a_388_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_396_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_396_ == 0)
{
v___x_390_ = v___x_387_;
v_isShared_391_ = v_isSharedCheck_396_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_a_388_);
lean_dec(v___x_387_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_396_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_392_; lean_object* v___x_394_; 
lean_inc(v_ref_386_);
v___x_392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_392_, 0, v_ref_386_);
lean_ctor_set(v___x_392_, 1, v_a_388_);
if (v_isShared_391_ == 0)
{
lean_ctor_set_tag(v___x_390_, 1);
lean_ctor_set(v___x_390_, 0, v___x_392_);
v___x_394_ = v___x_390_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v___x_392_);
v___x_394_ = v_reuseFailAlloc_395_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
return v___x_394_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v_msg_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_){
_start:
{
lean_object* v_res_401_; 
v_res_401_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg(v_msg_397_, v___y_398_, v___y_399_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
return v_res_401_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_403_; lean_object* v___x_404_; 
v___x_403_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_));
v___x_404_ = l_Lean_stringToMessageData(v___x_403_);
return v___x_404_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_406_; lean_object* v___x_407_; 
v___x_406_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_));
v___x_407_ = l_Lean_stringToMessageData(v___x_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(lean_object* v___x_408_, lean_object* v___x_409_, lean_object* v___x_410_, lean_object* v___x_411_, lean_object* v___x_412_, lean_object* v_declName_413_, lean_object* v_stx_414_, uint8_t v_kind_415_, lean_object* v___y_416_, lean_object* v___y_417_){
_start:
{
lean_object* v___y_420_; lean_object* v___y_421_; lean_object* v___y_422_; lean_object* v___x_448_; uint8_t v___x_449_; lean_object* v___y_451_; lean_object* v___y_452_; lean_object* v___y_453_; lean_object* v___y_465_; lean_object* v___y_466_; lean_object* v___y_467_; lean_object* v___y_471_; lean_object* v___y_472_; 
v___x_448_ = l_Lean_Name_mkStr4(v___x_408_, v___x_409_, v___x_410_, v___x_411_);
v___x_449_ = l_Lean_Syntax_isOfKind(v_stx_414_, v___x_448_);
lean_dec(v___x_448_);
if (v___x_449_ == 0)
{
lean_object* v___x_476_; 
lean_dec(v_declName_413_);
lean_dec(v___x_412_);
v___x_476_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__2___redArg();
return v___x_476_;
}
else
{
uint8_t v___x_477_; uint8_t v___x_478_; 
v___x_477_ = 0;
v___x_478_ = l_Lean_instBEqAttributeKind_beq(v_kind_415_, v___x_477_);
if (v___x_478_ == 0)
{
lean_object* v___x_479_; lean_object* v___x_480_; 
lean_dec(v_declName_413_);
lean_dec(v___x_412_);
v___x_479_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_);
v___x_480_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg(v___x_479_, v___y_416_, v___y_417_);
return v___x_480_;
}
else
{
v___y_471_ = v___y_416_;
v___y_472_ = v___y_417_;
goto v___jp_470_;
}
}
v___jp_419_:
{
lean_object* v___x_423_; lean_object* v_env_424_; lean_object* v_options_425_; lean_object* v_ref_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_423_ = lean_st_ref_get(v___y_422_);
v_env_424_ = lean_ctor_get(v___x_423_, 0);
lean_inc_ref(v_env_424_);
lean_dec(v___x_423_);
v_options_425_ = lean_ctor_get(v___y_421_, 2);
v_ref_426_ = lean_ctor_get(v___y_421_, 5);
lean_inc_ref(v_options_425_);
v___x_427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_427_, 0, v_env_424_);
lean_ctor_set(v___x_427_, 1, v_options_425_);
lean_inc(v_declName_413_);
v___x_428_ = lp_mathlib_Mathlib_Tactic_Polynomial_mkPolynomialExt(v_declName_413_, v___x_427_);
lean_dec_ref_known(v___x_427_, 2);
if (lean_obj_tag(v___x_428_) == 0)
{
lean_object* v_a_429_; lean_object* v___x_430_; lean_object* v_toEnvExtension_431_; lean_object* v_asyncMode_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; 
v_a_429_ = lean_ctor_get(v___x_428_, 0);
lean_inc(v_a_429_);
lean_dec_ref_known(v___x_428_, 1);
v___x_430_ = lp_mathlib_Mathlib_Tactic_Polynomial_polynomialExt;
v_toEnvExtension_431_ = lean_ctor_get(v___x_430_, 0);
v_asyncMode_432_ = lean_ctor_get(v_toEnvExtension_431_, 2);
v___x_433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_433_, 0, v_declName_413_);
lean_ctor_set(v___x_433_, 1, v_a_429_);
v___x_434_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_430_, v___y_420_, v___x_433_, v_asyncMode_432_, v___x_412_);
v___x_435_ = lp_mathlib_Lean_setEnv___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__0___redArg(v___x_434_, v___y_422_);
return v___x_435_;
}
else
{
lean_object* v_a_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_447_; 
lean_dec_ref(v___y_420_);
lean_dec(v_declName_413_);
lean_dec(v___x_412_);
v_a_436_ = lean_ctor_get(v___x_428_, 0);
v_isSharedCheck_447_ = !lean_is_exclusive(v___x_428_);
if (v_isSharedCheck_447_ == 0)
{
v___x_438_ = v___x_428_;
v_isShared_439_ = v_isSharedCheck_447_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_a_436_);
lean_dec(v___x_428_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_447_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_445_; 
v___x_440_ = lean_io_error_to_string(v_a_436_);
v___x_441_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_441_, 0, v___x_440_);
v___x_442_ = l_Lean_MessageData_ofFormat(v___x_441_);
lean_inc(v_ref_426_);
v___x_443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_443_, 0, v_ref_426_);
lean_ctor_set(v___x_443_, 1, v___x_442_);
if (v_isShared_439_ == 0)
{
lean_ctor_set(v___x_438_, 0, v___x_443_);
v___x_445_ = v___x_438_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_446_; 
v_reuseFailAlloc_446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_446_, 0, v___x_443_);
v___x_445_ = v_reuseFailAlloc_446_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
return v___x_445_;
}
}
}
}
v___jp_450_:
{
lean_object* v___x_454_; 
lean_inc(v_declName_413_);
lean_inc_ref(v___y_451_);
v___x_454_ = lean_decl_get_sorry_dep(v___y_451_, v_declName_413_);
if (lean_obj_tag(v___x_454_) == 0)
{
v___y_420_ = v___y_451_;
v___y_421_ = v___y_452_;
v___y_422_ = v___y_453_;
goto v___jp_419_;
}
else
{
lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_462_; 
v_isSharedCheck_462_ = !lean_is_exclusive(v___x_454_);
if (v_isSharedCheck_462_ == 0)
{
lean_object* v_unused_463_; 
v_unused_463_ = lean_ctor_get(v___x_454_, 0);
lean_dec(v_unused_463_);
v___x_456_ = v___x_454_;
v_isShared_457_ = v_isSharedCheck_462_;
goto v_resetjp_455_;
}
else
{
lean_dec(v___x_454_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_462_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
if (v___x_449_ == 0)
{
lean_del_object(v___x_456_);
v___y_420_ = v___y_451_;
v___y_421_ = v___y_452_;
v___y_422_ = v___y_453_;
goto v___jp_419_;
}
else
{
lean_object* v___x_458_; lean_object* v___x_460_; 
lean_dec_ref(v___y_451_);
lean_dec(v_declName_413_);
lean_dec(v___x_412_);
v___x_458_ = lean_box(0);
if (v_isShared_457_ == 0)
{
lean_ctor_set_tag(v___x_456_, 0);
lean_ctor_set(v___x_456_, 0, v___x_458_);
v___x_460_ = v___x_456_;
goto v_reusejp_459_;
}
else
{
lean_object* v_reuseFailAlloc_461_; 
v_reuseFailAlloc_461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_461_, 0, v___x_458_);
v___x_460_ = v_reuseFailAlloc_461_;
goto v_reusejp_459_;
}
v_reusejp_459_:
{
return v___x_460_;
}
}
}
}
}
v___jp_464_:
{
lean_object* v___x_468_; lean_object* v___x_469_; 
lean_dec_ref(v___y_465_);
v___x_468_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_);
v___x_469_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg(v___x_468_, v___y_467_, v___y_466_);
return v___x_469_;
}
v___jp_470_:
{
lean_object* v___x_473_; lean_object* v_env_474_; lean_object* v___x_475_; 
v___x_473_ = lean_st_ref_get(v___y_472_);
v_env_474_ = lean_ctor_get(v___x_473_, 0);
lean_inc_ref(v_env_474_);
lean_dec(v___x_473_);
v___x_475_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_474_, v_declName_413_);
if (lean_obj_tag(v___x_475_) == 0)
{
if (v___x_449_ == 0)
{
lean_dec(v_declName_413_);
lean_dec(v___x_412_);
v___y_465_ = v_env_474_;
v___y_466_ = v___y_472_;
v___y_467_ = v___y_471_;
goto v___jp_464_;
}
else
{
v___y_451_ = v_env_474_;
v___y_452_ = v___y_471_;
v___y_453_ = v___y_472_;
goto v___jp_450_;
}
}
else
{
lean_dec_ref_known(v___x_475_, 1);
lean_dec(v_declName_413_);
lean_dec(v___x_412_);
v___y_465_ = v_env_474_;
v___y_466_ = v___y_472_;
v___y_467_ = v___y_471_;
goto v___jp_464_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2____boxed(lean_object* v___x_481_, lean_object* v___x_482_, lean_object* v___x_483_, lean_object* v___x_484_, lean_object* v___x_485_, lean_object* v_declName_486_, lean_object* v_stx_487_, lean_object* v_kind_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_){
_start:
{
uint8_t v_kind_boxed_492_; lean_object* v_res_493_; 
v_kind_boxed_492_ = lean_unbox(v_kind_488_);
v_res_493_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(v___x_481_, v___x_482_, v___x_483_, v___x_484_, v___x_485_, v_declName_486_, v_stx_487_, v_kind_boxed_492_, v___y_489_, v___y_490_);
lean_dec(v___y_490_);
lean_dec_ref(v___y_489_);
return v_res_493_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_495_; lean_object* v___x_496_; 
v___x_495_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_));
v___x_496_ = l_Lean_stringToMessageData(v___x_495_);
return v___x_496_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_498_; lean_object* v___x_499_; 
v___x_498_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_));
v___x_499_ = l_Lean_stringToMessageData(v___x_498_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(lean_object* v___x_500_, lean_object* v_decl_501_, lean_object* v___y_502_, lean_object* v___y_503_){
_start:
{
lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; 
v___x_505_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_);
v___x_506_ = l_Lean_MessageData_ofName(v___x_500_);
v___x_507_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_507_, 0, v___x_505_);
lean_ctor_set(v___x_507_, 1, v___x_506_);
v___x_508_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_);
v___x_509_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_509_, 0, v___x_507_);
lean_ctor_set(v___x_509_, 1, v___x_508_);
v___x_510_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg(v___x_509_, v___y_502_, v___y_503_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2____boxed(lean_object* v___x_511_, lean_object* v_decl_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___lam__1_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(v___x_511_, v_decl_512_, v___y_513_, v___y_514_);
lean_dec(v___y_514_);
lean_dec_ref(v___y_513_);
lean_dec(v_decl_512_);
return v_res_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_601_; lean_object* v___x_602_; 
v___x_601_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn___closed__30_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_));
v___x_602_ = l_Lean_registerBuiltinAttribute(v___x_601_);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2____boxed(lean_object* v_a_603_){
_start:
{
lean_object* v_res_604_; 
v_res_604_ = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_();
return v_res_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_605_, lean_object* v_msg_606_, lean_object* v___y_607_, lean_object* v___y_608_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___redArg(v_msg_606_, v___y_607_, v___y_608_);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_611_, lean_object* v_msg_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_){
_start:
{
lean_object* v_res_616_; 
v_res_616_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2__spec__1(v_00_u03b1_611_, v_msg_612_, v___y_613_, v___y_614_);
lean_dec(v___y_614_);
lean_dec_ref(v___y_613_);
return v_res_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1_spec__1(lean_object* v_msgData_617_, lean_object* v___y_618_, lean_object* v___y_619_, lean_object* v___y_620_, lean_object* v___y_621_){
_start:
{
lean_object* v___x_623_; lean_object* v_env_624_; lean_object* v___x_625_; lean_object* v_mctx_626_; lean_object* v_lctx_627_; lean_object* v_options_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_623_ = lean_st_ref_get(v___y_621_);
v_env_624_ = lean_ctor_get(v___x_623_, 0);
lean_inc_ref(v_env_624_);
lean_dec(v___x_623_);
v___x_625_ = lean_st_ref_get(v___y_619_);
v_mctx_626_ = lean_ctor_get(v___x_625_, 0);
lean_inc_ref(v_mctx_626_);
lean_dec(v___x_625_);
v_lctx_627_ = lean_ctor_get(v___y_618_, 2);
v_options_628_ = lean_ctor_get(v___y_620_, 2);
lean_inc_ref(v_options_628_);
lean_inc_ref(v_lctx_627_);
v___x_629_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_629_, 0, v_env_624_);
lean_ctor_set(v___x_629_, 1, v_mctx_626_);
lean_ctor_set(v___x_629_, 2, v_lctx_627_);
lean_ctor_set(v___x_629_, 3, v_options_628_);
v___x_630_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_630_, 0, v___x_629_);
lean_ctor_set(v___x_630_, 1, v_msgData_617_);
v___x_631_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_631_, 0, v___x_630_);
return v___x_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1_spec__1___boxed(lean_object* v_msgData_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_){
_start:
{
lean_object* v_res_638_; 
v_res_638_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1_spec__1(v_msgData_632_, v___y_633_, v___y_634_, v___y_635_, v___y_636_);
lean_dec(v___y_636_);
lean_dec_ref(v___y_635_);
lean_dec(v___y_634_);
lean_dec_ref(v___y_633_);
return v_res_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___redArg(lean_object* v_msg_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_){
_start:
{
lean_object* v_ref_645_; lean_object* v___x_646_; lean_object* v_a_647_; lean_object* v___x_649_; uint8_t v_isShared_650_; uint8_t v_isSharedCheck_655_; 
v_ref_645_ = lean_ctor_get(v___y_642_, 5);
v___x_646_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1_spec__1(v_msg_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_);
v_a_647_ = lean_ctor_get(v___x_646_, 0);
v_isSharedCheck_655_ = !lean_is_exclusive(v___x_646_);
if (v_isSharedCheck_655_ == 0)
{
v___x_649_ = v___x_646_;
v_isShared_650_ = v_isSharedCheck_655_;
goto v_resetjp_648_;
}
else
{
lean_inc(v_a_647_);
lean_dec(v___x_646_);
v___x_649_ = lean_box(0);
v_isShared_650_ = v_isSharedCheck_655_;
goto v_resetjp_648_;
}
v_resetjp_648_:
{
lean_object* v___x_651_; lean_object* v___x_653_; 
lean_inc(v_ref_645_);
v___x_651_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_651_, 0, v_ref_645_);
lean_ctor_set(v___x_651_, 1, v_a_647_);
if (v_isShared_650_ == 0)
{
lean_ctor_set_tag(v___x_649_, 1);
lean_ctor_set(v___x_649_, 0, v___x_651_);
v___x_653_ = v___x_649_;
goto v_reusejp_652_;
}
else
{
lean_object* v_reuseFailAlloc_654_; 
v_reuseFailAlloc_654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_654_, 0, v___x_651_);
v___x_653_ = v_reuseFailAlloc_654_;
goto v_reusejp_652_;
}
v_reusejp_652_:
{
return v___x_653_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___redArg___boxed(lean_object* v_msg_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_){
_start:
{
lean_object* v_res_662_; 
v_res_662_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___redArg(v_msg_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
lean_dec(v___y_660_);
lean_dec_ref(v___y_659_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
return v_res_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg(lean_object* v_e_666_, lean_object* v_as_x27_667_, lean_object* v_b_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
if (lean_obj_tag(v_as_x27_667_) == 0)
{
lean_object* v___x_674_; 
lean_dec_ref(v_e_666_);
v___x_674_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_674_, 0, v_b_668_);
return v___x_674_;
}
else
{
lean_object* v_head_675_; lean_object* v_tail_676_; lean_object* v_snd_677_; lean_object* v___x_678_; lean_object* v___x_679_; 
lean_dec_ref(v_b_668_);
v_head_675_ = lean_ctor_get(v_as_x27_667_, 0);
v_tail_676_ = lean_ctor_get(v_as_x27_667_, 1);
v_snd_677_ = lean_ctor_get(v_head_675_, 1);
v___x_678_ = lean_box(0);
lean_inc(v_snd_677_);
lean_inc(v___y_672_);
lean_inc_ref(v___y_671_);
lean_inc(v___y_670_);
lean_inc_ref(v___y_669_);
lean_inc_ref(v_e_666_);
v___x_679_ = lean_apply_6(v_snd_677_, v_e_666_, v___y_669_, v___y_670_, v___y_671_, v___y_672_, lean_box(0));
if (lean_obj_tag(v___x_679_) == 0)
{
lean_object* v_a_680_; lean_object* v___x_682_; uint8_t v_isShared_683_; uint8_t v_isSharedCheck_689_; 
lean_dec_ref(v_e_666_);
v_a_680_ = lean_ctor_get(v___x_679_, 0);
v_isSharedCheck_689_ = !lean_is_exclusive(v___x_679_);
if (v_isSharedCheck_689_ == 0)
{
v___x_682_ = v___x_679_;
v_isShared_683_ = v_isSharedCheck_689_;
goto v_resetjp_681_;
}
else
{
lean_inc(v_a_680_);
lean_dec(v___x_679_);
v___x_682_ = lean_box(0);
v_isShared_683_ = v_isSharedCheck_689_;
goto v_resetjp_681_;
}
v_resetjp_681_:
{
lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_687_; 
v___x_684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_684_, 0, v_a_680_);
v___x_685_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_685_, 0, v___x_684_);
lean_ctor_set(v___x_685_, 1, v___x_678_);
if (v_isShared_683_ == 0)
{
lean_ctor_set(v___x_682_, 0, v___x_685_);
v___x_687_ = v___x_682_;
goto v_reusejp_686_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v___x_685_);
v___x_687_ = v_reuseFailAlloc_688_;
goto v_reusejp_686_;
}
v_reusejp_686_:
{
return v___x_687_;
}
}
}
else
{
lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_703_; 
v_a_690_ = lean_ctor_get(v___x_679_, 0);
v_isSharedCheck_703_ = !lean_is_exclusive(v___x_679_);
if (v_isSharedCheck_703_ == 0)
{
v___x_692_ = v___x_679_;
v_isShared_693_ = v_isSharedCheck_703_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_dec(v___x_679_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_703_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v___x_694_; uint8_t v___y_696_; uint8_t v___x_701_; 
v___x_694_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg___closed__0));
v___x_701_ = l_Lean_Exception_isInterrupt(v_a_690_);
if (v___x_701_ == 0)
{
uint8_t v___x_702_; 
lean_inc(v_a_690_);
v___x_702_ = l_Lean_Exception_isRuntime(v_a_690_);
v___y_696_ = v___x_702_;
goto v___jp_695_;
}
else
{
v___y_696_ = v___x_701_;
goto v___jp_695_;
}
v___jp_695_:
{
if (v___y_696_ == 0)
{
lean_del_object(v___x_692_);
lean_dec(v_a_690_);
v_as_x27_667_ = v_tail_676_;
v_b_668_ = v___x_694_;
goto _start;
}
else
{
lean_object* v___x_699_; 
lean_dec_ref(v_e_666_);
if (v_isShared_693_ == 0)
{
v___x_699_ = v___x_692_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v_a_690_);
v___x_699_ = v_reuseFailAlloc_700_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
return v___x_699_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg___boxed(lean_object* v_e_704_, lean_object* v_as_x27_705_, lean_object* v_b_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg(v_e_704_, v_as_x27_705_, v_b_706_, v___y_707_, v___y_708_, v___y_709_, v___y_710_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v___y_708_);
lean_dec_ref(v___y_707_);
lean_dec(v_as_x27_705_);
return v_res_712_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__2(void){
_start:
{
lean_object* v___x_716_; lean_object* v___x_717_; 
v___x_716_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__1));
v___x_717_ = l_Lean_stringToMessageData(v___x_716_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_inferBase(lean_object* v_e_718_, lean_object* v_a_719_, lean_object* v_a_720_, lean_object* v_a_721_, lean_object* v_a_722_){
_start:
{
lean_object* v___x_724_; lean_object* v_env_725_; lean_object* v___x_726_; lean_object* v_toEnvExtension_727_; lean_object* v_asyncMode_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v_snd_732_; lean_object* v___x_733_; lean_object* v___x_734_; 
v___x_724_ = lean_st_ref_get(v_a_722_);
v_env_725_ = lean_ctor_get(v___x_724_, 0);
lean_inc_ref(v_env_725_);
lean_dec(v___x_724_);
v___x_726_ = lp_mathlib_Mathlib_Tactic_Polynomial_polynomialExt;
v_toEnvExtension_727_ = lean_ctor_get(v___x_726_, 0);
v_asyncMode_728_ = lean_ctor_get(v_toEnvExtension_727_, 2);
v___x_729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__0));
v___x_730_ = lean_box(0);
v___x_731_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_729_, v___x_726_, v_env_725_, v_asyncMode_728_, v___x_730_);
v_snd_732_ = lean_ctor_get(v___x_731_, 1);
lean_inc(v_snd_732_);
lean_dec(v___x_731_);
v___x_733_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg___closed__0));
v___x_734_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg(v_e_718_, v_snd_732_, v___x_733_, v_a_719_, v_a_720_, v_a_721_, v_a_722_);
lean_dec(v_snd_732_);
if (lean_obj_tag(v___x_734_) == 0)
{
lean_object* v_a_735_; lean_object* v___x_737_; uint8_t v_isShared_738_; uint8_t v_isSharedCheck_746_; 
v_a_735_ = lean_ctor_get(v___x_734_, 0);
v_isSharedCheck_746_ = !lean_is_exclusive(v___x_734_);
if (v_isSharedCheck_746_ == 0)
{
v___x_737_ = v___x_734_;
v_isShared_738_ = v_isSharedCheck_746_;
goto v_resetjp_736_;
}
else
{
lean_inc(v_a_735_);
lean_dec(v___x_734_);
v___x_737_ = lean_box(0);
v_isShared_738_ = v_isSharedCheck_746_;
goto v_resetjp_736_;
}
v_resetjp_736_:
{
lean_object* v_fst_739_; 
v_fst_739_ = lean_ctor_get(v_a_735_, 0);
lean_inc(v_fst_739_);
lean_dec(v_a_735_);
if (lean_obj_tag(v_fst_739_) == 0)
{
lean_object* v___x_740_; lean_object* v___x_741_; 
lean_del_object(v___x_737_);
v___x_740_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__2, &lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___closed__2);
v___x_741_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___redArg(v___x_740_, v_a_719_, v_a_720_, v_a_721_, v_a_722_);
return v___x_741_;
}
else
{
lean_object* v_val_742_; lean_object* v___x_744_; 
v_val_742_ = lean_ctor_get(v_fst_739_, 0);
lean_inc(v_val_742_);
lean_dec_ref_known(v_fst_739_, 1);
if (v_isShared_738_ == 0)
{
lean_ctor_set(v___x_737_, 0, v_val_742_);
v___x_744_ = v___x_737_;
goto v_reusejp_743_;
}
else
{
lean_object* v_reuseFailAlloc_745_; 
v_reuseFailAlloc_745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_745_, 0, v_val_742_);
v___x_744_ = v_reuseFailAlloc_745_;
goto v_reusejp_743_;
}
v_reusejp_743_:
{
return v___x_744_;
}
}
}
}
else
{
lean_object* v_a_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_754_; 
v_a_747_ = lean_ctor_get(v___x_734_, 0);
v_isSharedCheck_754_ = !lean_is_exclusive(v___x_734_);
if (v_isSharedCheck_754_ == 0)
{
v___x_749_ = v___x_734_;
v_isShared_750_ = v_isSharedCheck_754_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_a_747_);
lean_dec(v___x_734_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_754_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___x_752_; 
if (v_isShared_750_ == 0)
{
v___x_752_ = v___x_749_;
goto v_reusejp_751_;
}
else
{
lean_object* v_reuseFailAlloc_753_; 
v_reuseFailAlloc_753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_753_, 0, v_a_747_);
v___x_752_ = v_reuseFailAlloc_753_;
goto v_reusejp_751_;
}
v_reusejp_751_:
{
return v___x_752_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Polynomial_inferBase___boxed(lean_object* v_e_755_, lean_object* v_a_756_, lean_object* v_a_757_, lean_object* v_a_758_, lean_object* v_a_759_, lean_object* v_a_760_){
_start:
{
lean_object* v_res_761_; 
v_res_761_ = lp_mathlib_Mathlib_Tactic_Polynomial_inferBase(v_e_755_, v_a_756_, v_a_757_, v_a_758_, v_a_759_);
lean_dec(v_a_759_);
lean_dec_ref(v_a_758_);
lean_dec(v_a_757_);
lean_dec_ref(v_a_756_);
return v_res_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0(lean_object* v_e_762_, lean_object* v_as_763_, lean_object* v_as_x27_764_, lean_object* v_b_765_, lean_object* v_a_766_, lean_object* v___y_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___redArg(v_e_762_, v_as_x27_764_, v_b_765_, v___y_767_, v___y_768_, v___y_769_, v___y_770_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0___boxed(lean_object* v_e_773_, lean_object* v_as_774_, lean_object* v_as_x27_775_, lean_object* v_b_776_, lean_object* v_a_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_){
_start:
{
lean_object* v_res_783_; 
v_res_783_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Tactic_Polynomial_inferBase_spec__0(v_e_773_, v_as_774_, v_as_x27_775_, v_b_776_, v_a_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_);
lean_dec(v___y_781_);
lean_dec_ref(v___y_780_);
lean_dec(v___y_779_);
lean_dec_ref(v___y_778_);
lean_dec(v_as_x27_775_);
lean_dec(v_as_774_);
return v_res_783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1(lean_object* v_00_u03b1_784_, lean_object* v_msg_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_){
_start:
{
lean_object* v___x_791_; 
v___x_791_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___redArg(v_msg_785_, v___y_786_, v___y_787_, v___y_788_, v___y_789_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1___boxed(lean_object* v_00_u03b1_792_, lean_object* v_msg_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Polynomial_inferBase_spec__1(v_00_u03b1_792_, v_msg_793_, v___y_794_, v___y_795_, v___y_796_, v___y_797_);
lean_dec(v___y_797_);
lean_dec_ref(v___y_796_);
lean_dec(v___y_795_);
lean_dec_ref(v___y_794_);
return v_res_799_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Polynomial_Core(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Compiler_IR_CompilerM(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Attr(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Polynomial_Core(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Compiler_IR_CompilerM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_131907669____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Polynomial_polynomialPreExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Polynomial_polynomialPreExt);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_938055999____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Polynomial_polynomialPostExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Polynomial_polynomialPostExt);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_2526669815____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Polynomial_polynomialExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Polynomial_polynomialExt);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Polynomial_Core_0__Mathlib_Tactic_Polynomial_initFn_00___x40_Mathlib_Tactic_Polynomial_Core_737497776____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Compiler_IR_CompilerM(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Polynomial_Core(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Compiler_IR_CompilerM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Polynomial_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Polynomial_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Polynomial_Core(builtin);
}
#ifdef __cplusplus
}
#endif
