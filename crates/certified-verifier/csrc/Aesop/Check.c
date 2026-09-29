// Lean compiler output
// Module: Aesop.Check
// Imports: public import Init public meta import Init public import Lean.Data.Options
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Meta_Defs_0__Lean_getEscapedNameParts_x3f(lean_object*, lean_object*);
lean_object* l_Lean_quoteNameMk(lean_object*);
lean_object* lean_string_intercalate(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_mkNameLit(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "check"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "all"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 218, 127, 202, 106, 34, 40, 220)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(80, 49, 3, 137, 107, 57, 219, 91)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 75, .m_data = "(aesop) Enable all runtime checks. Individual checks can still be disabled."};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Check"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(208, 230, 253, 138, 118, 228, 177, 115)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(192, 131, 139, 206, 169, 30, 26, 20)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_all;
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get_x3f___at___00Aesop_Check_get_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get_x3f___at___00Aesop_Check_get_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Check_get_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Check_get_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Check_get(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_get___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_name(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_name___boxed(lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__1_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__1_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(213, 96, 250, 13, 195, 1, 48, 100)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__2_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(244, 177, 58, 253, 225, 61, 249, 210)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__3_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(117, 185, 155, 252, 231, 51, 236, 232)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__4_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__4_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(79, 68, 138, 14, 44, 32, 88, 55)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "commandRegister_aesop_check_option__"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__6_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__5_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(202, 166, 104, 136, 200, 134, 156, 206)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__7_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__8_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__9_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "register_aesop_check_option"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__10_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__10_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__11_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__12_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__13_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__13_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__14 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__14_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__9_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__11_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__14_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__15 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__15_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__16 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__16_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__16_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__17 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__17_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__17_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__18 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__18_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__9_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__15_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__18_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__19 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__19_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__7_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__19_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__20 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__20_value;
LEAN_EXPORT const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option____ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__20_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "initialize"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__5_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(55, 206, 156, 211, 241, 221, 187, 166)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__7_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__9;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "initializeKeyword"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__10_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(113, 140, 114, 135, 71, 133, 96, 5)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "option"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__12_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__13;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(86, 190, 205, 176, 185, 187, 129, 81)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__14 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__14_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__16 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__16_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__18 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__18_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__19 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__19_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Lean.Option"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__21 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__21_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__22;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Option"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__23 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__23_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__24_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(54, 183, 132, 140, 253, 175, 101, 43)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__24 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__24_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__25 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__25_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__24_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__26 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__26_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__27 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__27_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__25_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__27_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__28 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__28_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__29 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__29_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__30;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__31 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__31_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__31_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__32 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__32_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__31_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__33 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__33_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__34 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__34_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__32_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__34_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__35 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__35_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__36 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__36_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "doSeqIndent"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__37 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__37_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__37_value),LEAN_SCALAR_PTR_LITERAL(93, 115, 138, 230, 225, 195, 43, 46)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "doSeqItem"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__39 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__39_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__39_value),LEAN_SCALAR_PTR_LITERAL(10, 94, 50, 120, 46, 251, 13, 13)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "doExpr"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__41 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__41_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__41_value),LEAN_SCALAR_PTR_LITERAL(130, 168, 60, 255, 153, 218, 88, 77)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Lean.Option.register"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__43 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__43_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__44;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "register"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__45 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__45_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(54, 183, 132, 140, 253, 175, 101, 43)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__45_value),LEAN_SCALAR_PTR_LITERAL(127, 81, 22, 2, 70, 205, 7, 158)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__47 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__47_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__47_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__48 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__48_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__49_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 218, 127, 202, 106, 34, 40, 220)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__49 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__49_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "structInst"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__50 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__50_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__50_value),LEAN_SCALAR_PTR_LITERAL(50, 43, 73, 62, 118, 124, 31, 28)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__52 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__52_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "structInstFields"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__53 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__53_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__53_value),LEAN_SCALAR_PTR_LITERAL(0, 82, 141, 43, 62, 171, 163, 69)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "structInstField"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__55 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__55_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__55_value),LEAN_SCALAR_PTR_LITERAL(50, 77, 20, 88, 28, 210, 230, 84)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "structInstLVal"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__57 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__57_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__57_value),LEAN_SCALAR_PTR_LITERAL(185, 133, 6, 147, 6, 183, 100, 198)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "defValue"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__59 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__59_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__60;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__59_value),LEAN_SCALAR_PTR_LITERAL(162, 134, 43, 84, 92, 210, 251, 95)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__61 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__61_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "structInstFieldDef"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__62 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__62_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__62_value),LEAN_SCALAR_PTR_LITERAL(81, 102, 39, 227, 176, 252, 65, 103)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__64 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__64_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__65 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__65_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__66;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__65_value),LEAN_SCALAR_PTR_LITERAL(160, 214, 196, 140, 104, 187, 164, 111)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__67 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__67_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__68_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__68_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__65_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__68 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__68_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__68_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__69 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__69_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__69_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__70 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__70_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__71 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__71_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "descr"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__72 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__72_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__73;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__72_value),LEAN_SCALAR_PTR_LITERAL(108, 60, 190, 73, 168, 166, 216, 62)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__74 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__74_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "optEllipsis"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__75 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__75_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__75_value),LEAN_SCALAR_PTR_LITERAL(13, 1, 242, 203, 207, 188, 181, 160)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__77 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__77_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__78 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__78_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__78_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "definition"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__80 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__80_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__80_value),LEAN_SCALAR_PTR_LITERAL(248, 187, 217, 228, 39, 184, 218, 135)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "def"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__82 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__82_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "declId"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__83 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__83_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__83_value),LEAN_SCALAR_PTR_LITERAL(243, 92, 136, 33, 216, 98, 92, 25)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(18, 212, 144, 219, 235, 2, 140, 52)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__85 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__85_value;
static const lean_array_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__86 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__86_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__1_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__86_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__87 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__87_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "optDeclSig"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__88 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__88_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__88_value),LEAN_SCALAR_PTR_LITERAL(26, 9, 103, 232, 183, 57, 246, 75)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Aesop.Check"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__90 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__90_value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__91_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__91;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__92_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__92_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(208, 230, 253, 138, 118, 228, 177, 115)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__92 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__92_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__92_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__93 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__93_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__92_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__94 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__94_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__94_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__95 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__95_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__93_value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__95_value)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__96 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__96_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declValSimple"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__97 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__97_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__97_value),LEAN_SCALAR_PTR_LITERAL(228, 117, 47, 248, 145, 185, 135, 188)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__99 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__99_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__99_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__101 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__101_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__102 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__102_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Termination"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__103 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__103_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__104_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "suffix"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__104 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__104_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__103_value),LEAN_SCALAR_PTR_LITERAL(128, 225, 226, 49, 186, 161, 212, 105)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__104_value),LEAN_SCALAR_PTR_LITERAL(245, 187, 99, 45, 217, 244, 244, 120)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "quotedName"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__106 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__106_value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107_value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__106_value),LEAN_SCALAR_PTR_LITERAL(217, 120, 158, 75, 195, 162, 2, 130)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__108 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__108_value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__109_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__109 = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__109_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "proofReconstruction"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 218, 127, 202, 106, 34, 40, 220)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(160, 95, 18, 129, 24, 207, 198, 251)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 67, .m_capacity = 67, .m_length = 66, .m_data = "(aesop) Typecheck partial proof terms during proof reconstruction."};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(116, 94, 49, 10, 176, 211, 83, 112)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(29, 49, 238, 245, 140, 26, 210, 128)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(87, 134, 154, 120, 82, 14, 71, 72)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__9_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(158, 69, 29, 205, 110, 151, 20, 171)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__9_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__9_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__10_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__10_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__11_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__11_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__11_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__12_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__12_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__13_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__13_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__13_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__14_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__14_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__15_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__15_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_option_00___x40_Aesop_Check_3841271143____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_proofReconstruction;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "tree"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 218, 127, 202, 106, 34, 40, 220)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(164, 88, 93, 112, 18, 93, 162, 42)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 96, .m_capacity = 96, .m_length = 95, .m_data = "(aesop) Check search tree invariants after every iteration of the search loop. Quite expensive."};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__9_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),((lean_object*)(((size_t)(445867947) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(22, 203, 251, 9, 222, 114, 30, 184)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__11_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(177, 226, 202, 199, 156, 23, 66, 141)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__13_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(73, 145, 97, 77, 163, 46, 58, 11)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(24, 130, 221, 212, 214, 5, 79, 206)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_445867947____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_445867947____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_option_00___x40_Aesop_Check_445867947____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_tree;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "rules"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 218, 127, 202, 106, 34, 40, 220)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(47, 188, 220, 151, 142, 112, 234, 60)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "(aesop) Check that information reported by rules is correct."};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__9_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),((lean_object*)(((size_t)(611402289) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(100, 207, 170, 137, 13, 157, 166, 45)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__11_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(203, 45, 127, 209, 99, 59, 247, 99)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__13_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(139, 47, 134, 33, 136, 104, 65, 113)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(2, 242, 120, 27, 226, 112, 11, 254)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_611402289____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_611402289____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_option_00___x40_Aesop_Check_611402289____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_rules;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "script"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 218, 127, 202, 106, 34, 40, 220)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(107, 8, 84, 236, 6, 154, 155, 88)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 170, .m_capacity = 170, .m_length = 169, .m_data = "(aesop) Check that the tactic script generated by Aesop proves the goal. When this check is active, Aesop generates a tactic script even if the user did not request one."};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__9_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),((lean_object*)(((size_t)(1191392668) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(238, 254, 145, 42, 88, 102, 198, 129)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__11_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(121, 94, 215, 177, 30, 49, 122, 53)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__13_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(49, 218, 12, 75, 188, 86, 129, 38)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(192, 119, 182, 178, 20, 115, 108, 66)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_option_00___x40_Aesop_Check_1191392668____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_script;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "steps"};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 218, 127, 202, 106, 34, 40, 220)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value_aux_1),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(107, 8, 84, 236, 6, 154, 155, 88)}};
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__0_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(168, 146, 148, 211, 245, 60, 94, 230)}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value;
static const lean_string_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 162, .m_capacity = 162, .m_length = 161, .m_data = "(aesop) Check each step of the tactic script generated by Aesop. When this check is active, Aesop generates a tactic script even if the user did not request one."};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__2_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__value;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_;
static lean_once_cell_t lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_option_00___x40_Aesop_Check_2536445200____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_script_steps;
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
_start:
{
lean_object* v_defValue_5_; lean_object* v_descr_6_; lean_object* v_deprecation_x3f_7_; lean_object* v___x_8_; uint8_t v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v_defValue_5_ = lean_ctor_get(v_decl_2_, 0);
v_descr_6_ = lean_ctor_get(v_decl_2_, 1);
v_deprecation_x3f_7_ = lean_ctor_get(v_decl_2_, 2);
v___x_8_ = lean_alloc_ctor(1, 0, 1);
v___x_9_ = lean_unbox(v_defValue_5_);
lean_ctor_set_uint8(v___x_8_, 0, v___x_9_);
lean_inc(v_deprecation_x3f_7_);
lean_inc_ref(v_descr_6_);
lean_inc_n(v_name_1_, 2);
v___x_10_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_10_, 0, v_name_1_);
lean_ctor_set(v___x_10_, 1, v_ref_3_);
lean_ctor_set(v___x_10_, 2, v___x_8_);
lean_ctor_set(v___x_10_, 3, v_descr_6_);
lean_ctor_set(v___x_10_, 4, v_deprecation_x3f_7_);
v___x_11_ = lean_register_option(v_name_1_, v___x_10_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_19_; 
v_isSharedCheck_19_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_19_ == 0)
{
lean_object* v_unused_20_; 
v_unused_20_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_20_);
v___x_13_ = v___x_11_;
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
else
{
lean_dec(v___x_11_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_17_; 
lean_inc(v_defValue_5_);
v___x_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_15_, 0, v_name_1_);
lean_ctor_set(v___x_15_, 1, v_defValue_5_);
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v___x_15_);
v___x_17_ = v___x_13_;
goto v_reusejp_16_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v___x_15_);
v___x_17_ = v_reuseFailAlloc_18_;
goto v_reusejp_16_;
}
v_reusejp_16_:
{
return v___x_17_;
}
}
}
else
{
lean_object* v_a_21_; lean_object* v___x_23_; uint8_t v_isShared_24_; uint8_t v_isSharedCheck_28_; 
lean_dec(v_name_1_);
v_a_21_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_28_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_28_ == 0)
{
v___x_23_ = v___x_11_;
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
else
{
lean_inc(v_a_21_);
lean_dec(v___x_11_);
v___x_23_ = lean_box(0);
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
v_resetjp_22_:
{
lean_object* v___x_26_; 
if (v_isShared_24_ == 0)
{
v___x_26_ = v___x_23_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v_a_21_);
v___x_26_ = v_reuseFailAlloc_27_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
return v___x_26_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_54_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_));
v___x_55_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_));
v___x_56_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__8_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_));
v___x_57_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(v___x_54_, v___x_55_, v___x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2____boxed(lean_object* v_a_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_();
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get_x3f___at___00Aesop_Check_get_spec__0(lean_object* v_opts_60_, lean_object* v_opt_61_){
_start:
{
lean_object* v_name_62_; lean_object* v_map_63_; lean_object* v___x_64_; 
v_name_62_ = lean_ctor_get(v_opt_61_, 0);
v_map_63_ = lean_ctor_get(v_opts_60_, 0);
v___x_64_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_63_, v_name_62_);
if (lean_obj_tag(v___x_64_) == 0)
{
lean_object* v___x_65_; 
v___x_65_ = lean_box(0);
return v___x_65_;
}
else
{
lean_object* v_val_66_; lean_object* v___x_68_; uint8_t v_isShared_69_; uint8_t v_isSharedCheck_76_; 
v_val_66_ = lean_ctor_get(v___x_64_, 0);
v_isSharedCheck_76_ = !lean_is_exclusive(v___x_64_);
if (v_isSharedCheck_76_ == 0)
{
v___x_68_ = v___x_64_;
v_isShared_69_ = v_isSharedCheck_76_;
goto v_resetjp_67_;
}
else
{
lean_inc(v_val_66_);
lean_dec(v___x_64_);
v___x_68_ = lean_box(0);
v_isShared_69_ = v_isSharedCheck_76_;
goto v_resetjp_67_;
}
v_resetjp_67_:
{
if (lean_obj_tag(v_val_66_) == 1)
{
uint8_t v_v_70_; lean_object* v___x_71_; lean_object* v___x_73_; 
v_v_70_ = lean_ctor_get_uint8(v_val_66_, 0);
lean_dec_ref_known(v_val_66_, 0);
v___x_71_ = lean_box(v_v_70_);
if (v_isShared_69_ == 0)
{
lean_ctor_set(v___x_68_, 0, v___x_71_);
v___x_73_ = v___x_68_;
goto v_reusejp_72_;
}
else
{
lean_object* v_reuseFailAlloc_74_; 
v_reuseFailAlloc_74_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_74_, 0, v___x_71_);
v___x_73_ = v_reuseFailAlloc_74_;
goto v_reusejp_72_;
}
v_reusejp_72_:
{
return v___x_73_;
}
}
else
{
lean_object* v___x_75_; 
lean_del_object(v___x_68_);
lean_dec(v_val_66_);
v___x_75_ = lean_box(0);
return v___x_75_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get_x3f___at___00Aesop_Check_get_spec__0___boxed(lean_object* v_opts_77_, lean_object* v_opt_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_aesop_Lean_Option_get_x3f___at___00Aesop_Check_get_spec__0(v_opts_77_, v_opt_78_);
lean_dec_ref(v_opt_78_);
lean_dec_ref(v_opts_77_);
return v_res_79_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Check_get_spec__1(lean_object* v_opts_80_, lean_object* v_opt_81_){
_start:
{
lean_object* v_name_82_; lean_object* v_defValue_83_; lean_object* v_map_84_; lean_object* v___x_85_; 
v_name_82_ = lean_ctor_get(v_opt_81_, 0);
v_defValue_83_ = lean_ctor_get(v_opt_81_, 1);
v_map_84_ = lean_ctor_get(v_opts_80_, 0);
v___x_85_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_84_, v_name_82_);
if (lean_obj_tag(v___x_85_) == 0)
{
uint8_t v___x_86_; 
v___x_86_ = lean_unbox(v_defValue_83_);
return v___x_86_;
}
else
{
lean_object* v_val_87_; 
v_val_87_ = lean_ctor_get(v___x_85_, 0);
lean_inc(v_val_87_);
lean_dec_ref_known(v___x_85_, 1);
if (lean_obj_tag(v_val_87_) == 1)
{
uint8_t v_v_88_; 
v_v_88_ = lean_ctor_get_uint8(v_val_87_, 0);
lean_dec_ref_known(v_val_87_, 0);
return v_v_88_;
}
else
{
uint8_t v___x_89_; 
lean_dec(v_val_87_);
v___x_89_ = lean_unbox(v_defValue_83_);
return v___x_89_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Check_get_spec__1___boxed(lean_object* v_opts_90_, lean_object* v_opt_91_){
_start:
{
uint8_t v_res_92_; lean_object* v_r_93_; 
v_res_92_ = lp_aesop_Lean_Option_get___at___00Aesop_Check_get_spec__1(v_opts_90_, v_opt_91_);
lean_dec_ref(v_opt_91_);
lean_dec_ref(v_opts_90_);
v_r_93_ = lean_box(v_res_92_);
return v_r_93_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Check_get(lean_object* v_opts_94_, lean_object* v_opt_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_aesop_Lean_Option_get_x3f___at___00Aesop_Check_get_spec__0(v_opts_94_, v_opt_95_);
if (lean_obj_tag(v___x_96_) == 1)
{
lean_object* v_val_97_; uint8_t v___x_98_; 
v_val_97_ = lean_ctor_get(v___x_96_, 0);
lean_inc(v_val_97_);
lean_dec_ref_known(v___x_96_, 1);
v___x_98_ = lean_unbox(v_val_97_);
lean_dec(v_val_97_);
return v___x_98_;
}
else
{
lean_object* v___x_99_; uint8_t v___x_100_; 
lean_dec(v___x_96_);
v___x_99_ = lp_aesop_Aesop_Check_all;
v___x_100_ = lp_aesop_Lean_Option_get___at___00Aesop_Check_get_spec__1(v_opts_94_, v___x_99_);
return v___x_100_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_get___boxed(lean_object* v_opts_101_, lean_object* v_opt_102_){
_start:
{
uint8_t v_res_103_; lean_object* v_r_104_; 
v_res_103_ = lp_aesop_Aesop_Check_get(v_opts_101_, v_opt_102_);
lean_dec_ref(v_opt_102_);
lean_dec_ref(v_opts_101_);
v_r_104_ = lean_box(v_res_103_);
return v_r_104_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___redArg___lam__0(lean_object* v_opt_105_, lean_object* v_toPure_106_, lean_object* v_____do__lift_107_){
_start:
{
uint8_t v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_108_ = lp_aesop_Aesop_Check_get(v_____do__lift_107_, v_opt_105_);
v___x_109_ = lean_box(v___x_108_);
v___x_110_ = lean_apply_2(v_toPure_106_, lean_box(0), v___x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___redArg___lam__0___boxed(lean_object* v_opt_111_, lean_object* v_toPure_112_, lean_object* v_____do__lift_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_aesop_Aesop_Check_isEnabled___redArg___lam__0(v_opt_111_, v_toPure_112_, v_____do__lift_113_);
lean_dec_ref(v_____do__lift_113_);
lean_dec_ref(v_opt_111_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled___redArg(lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_opt_117_){
_start:
{
lean_object* v_toApplicative_118_; lean_object* v_toBind_119_; lean_object* v_toPure_120_; lean_object* v___f_121_; lean_object* v___x_122_; 
v_toApplicative_118_ = lean_ctor_get(v_inst_115_, 0);
lean_inc_ref(v_toApplicative_118_);
v_toBind_119_ = lean_ctor_get(v_inst_115_, 1);
lean_inc(v_toBind_119_);
lean_dec_ref(v_inst_115_);
v_toPure_120_ = lean_ctor_get(v_toApplicative_118_, 1);
lean_inc(v_toPure_120_);
lean_dec_ref(v_toApplicative_118_);
v___f_121_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Check_isEnabled___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_121_, 0, v_opt_117_);
lean_closure_set(v___f_121_, 1, v_toPure_120_);
v___x_122_ = lean_apply_4(v_toBind_119_, lean_box(0), lean_box(0), v_inst_116_, v___f_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_isEnabled(lean_object* v_m_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_opt_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_aesop_Aesop_Check_isEnabled___redArg(v_inst_124_, v_inst_125_, v_opt_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_name(lean_object* v_opt_128_){
_start:
{
lean_object* v_name_129_; 
v_name_129_ = lean_ctor_get(v_opt_128_, 0);
lean_inc(v_name_129_);
return v_name_129_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Check_name___boxed(lean_object* v_opt_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_aesop_Aesop_Check_name(v_opt_130_);
lean_dec_ref(v_opt_130_);
return v_res_131_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__9(void){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = l_Array_mkArray0(lean_box(0));
return v___x_199_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__13(void){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_207_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__12));
v___x_208_ = l_String_toRawSubstring_x27(v___x_207_);
return v___x_208_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__22(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_226_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__21));
v___x_227_ = l_String_toRawSubstring_x27(v___x_226_);
return v___x_227_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__30(void){
_start:
{
lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_244_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__29));
v___x_245_ = l_String_toRawSubstring_x27(v___x_244_);
return v___x_245_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__44(void){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_279_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__43));
v___x_280_ = l_String_toRawSubstring_x27(v___x_279_);
return v___x_280_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__60(void){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_321_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__59));
v___x_322_ = l_String_toRawSubstring_x27(v___x_321_);
return v___x_322_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__66(void){
_start:
{
lean_object* v___x_333_; lean_object* v___x_334_; 
v___x_333_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__65));
v___x_334_ = l_String_toRawSubstring_x27(v___x_333_);
return v___x_334_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__73(void){
_start:
{
lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_348_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__72));
v___x_349_ = l_String_toRawSubstring_x27(v___x_348_);
return v___x_349_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__91(void){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_393_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__90));
v___x_394_ = l_String_toRawSubstring_x27(v___x_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1(lean_object* v_x_438_, lean_object* v_a_439_, lean_object* v_a_440_){
_start:
{
lean_object* v___x_441_; uint8_t v___x_442_; 
v___x_441_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_commandRegister__aesop__check__option_____00__closed__7));
lean_inc(v_x_438_);
v___x_442_ = l_Lean_Syntax_isOfKind(v_x_438_, v___x_441_);
if (v___x_442_ == 0)
{
lean_object* v___x_443_; lean_object* v___x_444_; 
lean_dec(v_x_438_);
v___x_443_ = lean_box(1);
v___x_444_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_444_, 0, v___x_443_);
lean_ctor_set(v___x_444_, 1, v_a_440_);
return v___x_444_;
}
else
{
lean_object* v_quotContext_445_; lean_object* v_currMacroScope_446_; lean_object* v_ref_447_; lean_object* v___x_448_; lean_object* v_optName_449_; lean_object* v___x_450_; lean_object* v___x_451_; uint8_t v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___y_500_; lean_object* v___x_585_; lean_object* v___x_586_; 
v_quotContext_445_ = lean_ctor_get(v_a_439_, 1);
v_currMacroScope_446_ = lean_ctor_get(v_a_439_, 2);
v_ref_447_ = lean_ctor_get(v_a_439_, 5);
v___x_448_ = lean_unsigned_to_nat(1u);
v_optName_449_ = l_Lean_Syntax_getArg(v_x_438_, v___x_448_);
v___x_450_ = lean_unsigned_to_nat(2u);
v___x_451_ = l_Lean_Syntax_getArg(v_x_438_, v___x_450_);
lean_dec(v_x_438_);
v___x_452_ = 0;
v___x_453_ = l_Lean_SourceInfo_fromRef(v_ref_447_, v___x_452_);
v___x_454_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__1));
v___x_455_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__5));
v___x_456_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__6));
v___x_457_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__8));
v___x_458_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__9, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__9_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__9);
lean_inc_n(v___x_453_, 14);
v___x_459_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_459_, 0, v___x_453_);
lean_ctor_set(v___x_459_, 1, v___x_454_);
lean_ctor_set(v___x_459_, 2, v___x_458_);
lean_inc_ref_n(v___x_459_, 7);
v___x_460_ = l_Lean_Syntax_node7(v___x_453_, v___x_457_, v___x_459_, v___x_459_, v___x_459_, v___x_459_, v___x_459_, v___x_459_, v___x_459_);
v___x_461_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__11));
v___x_462_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_462_, 0, v___x_453_);
lean_ctor_set(v___x_462_, 1, v___x_455_);
v___x_463_ = l_Lean_Syntax_node1(v___x_453_, v___x_461_, v___x_462_);
v___x_464_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__13, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__13_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__13);
v___x_465_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__14));
lean_inc_n(v_currMacroScope_446_, 4);
lean_inc_n(v_quotContext_445_, 4);
v___x_466_ = l_Lean_addMacroScope(v_quotContext_445_, v___x_465_, v_currMacroScope_446_);
v___x_467_ = lean_box(0);
v___x_468_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_468_, 0, v___x_453_);
lean_ctor_set(v___x_468_, 1, v___x_464_);
lean_ctor_set(v___x_468_, 2, v___x_466_);
lean_ctor_set(v___x_468_, 3, v___x_467_);
v___x_469_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__17));
v___x_470_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__18));
v___x_471_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_471_, 0, v___x_453_);
lean_ctor_set(v___x_471_, 1, v___x_470_);
v___x_472_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__20));
v___x_473_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__22, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__22_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__22);
v___x_474_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__24));
v___x_475_ = l_Lean_addMacroScope(v_quotContext_445_, v___x_474_, v_currMacroScope_446_);
v___x_476_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__28));
v___x_477_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_477_, 0, v___x_453_);
lean_ctor_set(v___x_477_, 1, v___x_473_);
lean_ctor_set(v___x_477_, 2, v___x_475_);
lean_ctor_set(v___x_477_, 3, v___x_476_);
v___x_478_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__30, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__30_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__30);
v___x_479_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__31));
v___x_480_ = l_Lean_addMacroScope(v_quotContext_445_, v___x_479_, v_currMacroScope_446_);
v___x_481_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__35));
v___x_482_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_482_, 0, v___x_453_);
lean_ctor_set(v___x_482_, 1, v___x_478_);
lean_ctor_set(v___x_482_, 2, v___x_480_);
lean_ctor_set(v___x_482_, 3, v___x_481_);
v___x_483_ = l_Lean_Syntax_node1(v___x_453_, v___x_454_, v___x_482_);
v___x_484_ = l_Lean_Syntax_node2(v___x_453_, v___x_472_, v___x_477_, v___x_483_);
lean_inc_ref(v___x_471_);
v___x_485_ = l_Lean_Syntax_node2(v___x_453_, v___x_469_, v___x_471_, v___x_484_);
v___x_486_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__36));
v___x_487_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_487_, 0, v___x_453_);
lean_ctor_set(v___x_487_, 1, v___x_486_);
lean_inc_ref(v___x_468_);
v___x_488_ = l_Lean_Syntax_node3(v___x_453_, v___x_454_, v___x_468_, v___x_485_, v___x_487_);
v___x_489_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__38));
v___x_490_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__40));
v___x_491_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__42));
v___x_492_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__44, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__44_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__44);
v___x_493_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__46));
v___x_494_ = l_Lean_addMacroScope(v_quotContext_445_, v___x_493_, v_currMacroScope_446_);
v___x_495_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__48));
v___x_496_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_496_, 0, v___x_453_);
lean_ctor_set(v___x_496_, 1, v___x_492_);
lean_ctor_set(v___x_496_, 2, v___x_494_);
lean_ctor_set(v___x_496_, 3, v___x_495_);
v___x_497_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__49));
v___x_498_ = l_Lean_TSyntax_getId(v_optName_449_);
lean_dec(v_optName_449_);
lean_inc(v___x_498_);
v___x_585_ = l_Lean_Name_append(v___x_497_, v___x_498_);
lean_inc(v___x_585_);
v___x_586_ = l___private_Init_Meta_Defs_0__Lean_getEscapedNameParts_x3f(v___x_467_, v___x_585_);
if (lean_obj_tag(v___x_586_) == 0)
{
lean_object* v___x_587_; 
v___x_587_ = l_Lean_quoteNameMk(v___x_585_);
v___y_500_ = v___x_587_;
goto v___jp_499_;
}
else
{
lean_object* v_val_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; 
lean_dec(v___x_585_);
v_val_588_ = lean_ctor_get(v___x_586_, 0);
lean_inc(v_val_588_);
lean_dec_ref_known(v___x_586_, 1);
v___x_589_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__107));
v___x_590_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__108));
v___x_591_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__109));
v___x_592_ = lean_string_intercalate(v___x_591_, v_val_588_);
v___x_593_ = lean_string_append(v___x_590_, v___x_592_);
lean_dec_ref(v___x_592_);
v___x_594_ = lean_box(2);
v___x_595_ = l_Lean_Syntax_mkNameLit(v___x_593_, v___x_594_);
v___x_596_ = lean_mk_empty_array_with_capacity(v___x_448_);
v___x_597_ = lean_array_push(v___x_596_, v___x_595_);
v___x_598_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_598_, 0, v___x_594_);
lean_ctor_set(v___x_598_, 1, v___x_589_);
lean_ctor_set(v___x_598_, 2, v___x_597_);
v___y_500_ = v___x_598_;
goto v___jp_499_;
}
v___jp_499_:
{
lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; 
v___x_501_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__51));
v___x_502_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__52));
lean_inc_n(v___x_453_, 39);
v___x_503_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_453_);
lean_ctor_set(v___x_503_, 1, v___x_502_);
v___x_504_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__54));
v___x_505_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__56));
v___x_506_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__58));
v___x_507_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__60, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__60_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__60);
v___x_508_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__61));
lean_inc_n(v_currMacroScope_446_, 4);
lean_inc_n(v_quotContext_445_, 4);
v___x_509_ = l_Lean_addMacroScope(v_quotContext_445_, v___x_508_, v_currMacroScope_446_);
v___x_510_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_510_, 0, v___x_453_);
lean_ctor_set(v___x_510_, 1, v___x_507_);
lean_ctor_set(v___x_510_, 2, v___x_509_);
lean_ctor_set(v___x_510_, 3, v___x_467_);
lean_inc_ref_n(v___x_459_, 16);
v___x_511_ = l_Lean_Syntax_node2(v___x_453_, v___x_506_, v___x_510_, v___x_459_);
v___x_512_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__63));
v___x_513_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__64));
v___x_514_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_514_, 0, v___x_453_);
lean_ctor_set(v___x_514_, 1, v___x_513_);
v___x_515_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__66, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__66_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__66);
v___x_516_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__67));
v___x_517_ = l_Lean_addMacroScope(v_quotContext_445_, v___x_516_, v_currMacroScope_446_);
v___x_518_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__70));
v___x_519_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_519_, 0, v___x_453_);
lean_ctor_set(v___x_519_, 1, v___x_515_);
lean_ctor_set(v___x_519_, 2, v___x_517_);
lean_ctor_set(v___x_519_, 3, v___x_518_);
lean_inc_ref_n(v___x_514_, 2);
v___x_520_ = l_Lean_Syntax_node3(v___x_453_, v___x_512_, v___x_514_, v___x_459_, v___x_519_);
v___x_521_ = l_Lean_Syntax_node3(v___x_453_, v___x_454_, v___x_459_, v___x_459_, v___x_520_);
v___x_522_ = l_Lean_Syntax_node2(v___x_453_, v___x_505_, v___x_511_, v___x_521_);
v___x_523_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__71));
v___x_524_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_524_, 0, v___x_453_);
lean_ctor_set(v___x_524_, 1, v___x_523_);
v___x_525_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__73, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__73_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__73);
v___x_526_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__74));
v___x_527_ = l_Lean_addMacroScope(v_quotContext_445_, v___x_526_, v_currMacroScope_446_);
v___x_528_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_528_, 0, v___x_453_);
lean_ctor_set(v___x_528_, 1, v___x_525_);
lean_ctor_set(v___x_528_, 2, v___x_527_);
lean_ctor_set(v___x_528_, 3, v___x_467_);
v___x_529_ = l_Lean_Syntax_node2(v___x_453_, v___x_506_, v___x_528_, v___x_459_);
v___x_530_ = l_Lean_Syntax_node3(v___x_453_, v___x_512_, v___x_514_, v___x_459_, v___x_451_);
v___x_531_ = l_Lean_Syntax_node3(v___x_453_, v___x_454_, v___x_459_, v___x_459_, v___x_530_);
v___x_532_ = l_Lean_Syntax_node2(v___x_453_, v___x_505_, v___x_529_, v___x_531_);
v___x_533_ = l_Lean_Syntax_node3(v___x_453_, v___x_454_, v___x_522_, v___x_524_, v___x_532_);
v___x_534_ = l_Lean_Syntax_node1(v___x_453_, v___x_504_, v___x_533_);
v___x_535_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__76));
v___x_536_ = l_Lean_Syntax_node1(v___x_453_, v___x_535_, v___x_459_);
v___x_537_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__77));
v___x_538_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_538_, 0, v___x_453_);
lean_ctor_set(v___x_538_, 1, v___x_537_);
v___x_539_ = l_Lean_Syntax_node6(v___x_453_, v___x_501_, v___x_503_, v___x_459_, v___x_534_, v___x_536_, v___x_459_, v___x_538_);
v___x_540_ = l_Lean_Syntax_node2(v___x_453_, v___x_454_, v___y_500_, v___x_539_);
v___x_541_ = l_Lean_Syntax_node2(v___x_453_, v___x_472_, v___x_496_, v___x_540_);
v___x_542_ = l_Lean_Syntax_node1(v___x_453_, v___x_491_, v___x_541_);
v___x_543_ = l_Lean_Syntax_node2(v___x_453_, v___x_490_, v___x_542_, v___x_459_);
v___x_544_ = l_Lean_Syntax_node1(v___x_453_, v___x_454_, v___x_543_);
v___x_545_ = l_Lean_Syntax_node1(v___x_453_, v___x_489_, v___x_544_);
lean_inc(v___x_460_);
v___x_546_ = l_Lean_Syntax_node4(v___x_453_, v___x_456_, v___x_460_, v___x_463_, v___x_488_, v___x_545_);
v___x_547_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__79));
v___x_548_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__81));
v___x_549_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__82));
v___x_550_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_550_, 0, v___x_453_);
lean_ctor_set(v___x_550_, 1, v___x_549_);
v___x_551_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__84));
v___x_552_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__85));
v___x_553_ = l_Lean_Name_append(v___x_552_, v___x_498_);
v___x_554_ = l_Lean_mkIdent(v___x_553_);
v___x_555_ = lean_box(2);
v___x_556_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__87));
v___x_557_ = lean_mk_empty_array_with_capacity(v___x_450_);
v___x_558_ = lean_array_push(v___x_557_, v___x_554_);
v___x_559_ = lean_array_push(v___x_558_, v___x_556_);
v___x_560_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_560_, 0, v___x_555_);
lean_ctor_set(v___x_560_, 1, v___x_551_);
lean_ctor_set(v___x_560_, 2, v___x_559_);
v___x_561_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__89));
v___x_562_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__91, &lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__91_once, _init_lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__91);
v___x_563_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__92));
v___x_564_ = l_Lean_addMacroScope(v_quotContext_445_, v___x_563_, v_currMacroScope_446_);
v___x_565_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__96));
v___x_566_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_566_, 0, v___x_453_);
lean_ctor_set(v___x_566_, 1, v___x_562_);
lean_ctor_set(v___x_566_, 2, v___x_564_);
lean_ctor_set(v___x_566_, 3, v___x_565_);
v___x_567_ = l_Lean_Syntax_node2(v___x_453_, v___x_469_, v___x_471_, v___x_566_);
v___x_568_ = l_Lean_Syntax_node1(v___x_453_, v___x_454_, v___x_567_);
v___x_569_ = l_Lean_Syntax_node2(v___x_453_, v___x_561_, v___x_459_, v___x_568_);
v___x_570_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__98));
v___x_571_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__100));
v___x_572_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__101));
v___x_573_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_573_, 0, v___x_453_);
lean_ctor_set(v___x_573_, 1, v___x_572_);
v___x_574_ = l_Lean_Syntax_node1(v___x_453_, v___x_454_, v___x_468_);
v___x_575_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__102));
v___x_576_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_576_, 0, v___x_453_);
lean_ctor_set(v___x_576_, 1, v___x_575_);
v___x_577_ = l_Lean_Syntax_node3(v___x_453_, v___x_571_, v___x_573_, v___x_574_, v___x_576_);
v___x_578_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___closed__105));
v___x_579_ = l_Lean_Syntax_node2(v___x_453_, v___x_578_, v___x_459_, v___x_459_);
v___x_580_ = l_Lean_Syntax_node4(v___x_453_, v___x_570_, v___x_514_, v___x_577_, v___x_579_, v___x_459_);
v___x_581_ = l_Lean_Syntax_node5(v___x_453_, v___x_548_, v___x_550_, v___x_560_, v___x_569_, v___x_580_, v___x_459_);
v___x_582_ = l_Lean_Syntax_node2(v___x_453_, v___x_547_, v___x_460_, v___x_581_);
v___x_583_ = l_Lean_Syntax_node2(v___x_453_, v___x_454_, v___x_546_, v___x_582_);
v___x_584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_584_, 0, v___x_583_);
lean_ctor_set(v___x_584_, 1, v_a_440_);
return v___x_584_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1___boxed(lean_object* v_x_599_, lean_object* v_a_600_, lean_object* v_a_601_){
_start:
{
lean_object* v_res_602_; 
v_res_602_ = lp_aesop___private_Aesop_Check_0__Aesop___aux__Aesop__Check______macroRules____private__Aesop__Check__0__Aesop__commandRegister__aesop__check__option______1(v_x_599_, v_a_600_, v_a_601_);
lean_dec_ref(v_a_600_);
return v_res_602_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__10_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; 
v___x_630_ = lean_unsigned_to_nat(3841271143u);
v___x_631_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__9_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_));
v___x_632_ = l_Lean_Name_num___override(v___x_631_, v___x_630_);
return v___x_632_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__12_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; 
v___x_634_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__11_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_));
v___x_635_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__10_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_, &lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__10_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__once, _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__10_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_);
v___x_636_ = l_Lean_Name_str___override(v___x_635_, v___x_634_);
return v___x_636_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__14_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; 
v___x_638_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__13_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_));
v___x_639_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__12_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_, &lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__12_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__once, _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__12_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_);
v___x_640_ = l_Lean_Name_str___override(v___x_639_, v___x_638_);
return v___x_640_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__15_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; 
v___x_641_ = lean_unsigned_to_nat(3u);
v___x_642_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__14_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_, &lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__14_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__once, _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__14_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_);
v___x_643_ = l_Lean_Name_num___override(v___x_642_, v___x_641_);
return v___x_643_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; 
v___x_645_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_));
v___x_646_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_));
v___x_647_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__15_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_, &lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__15_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5__once, _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__15_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_);
v___x_648_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(v___x_645_, v___x_646_, v___x_647_);
return v___x_648_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5____boxed(lean_object* v_a_649_){
_start:
{
lean_object* v_res_650_; 
v_res_650_ = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_();
return v_res_650_;
}
}
static lean_object* _init_lp_aesop_Aesop_Check_proofReconstruction(void){
_start:
{
lean_object* v___x_651_; 
v___x_651_ = lp_aesop_Aesop_option_00___x40_Aesop_Check_3841271143____hygCtx___hyg_3_;
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; 
v___x_676_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_));
v___x_677_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_));
v___x_678_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_));
v___x_679_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(v___x_676_, v___x_677_, v___x_678_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_445867947____hygCtx___hyg_5____boxed(lean_object* v_a_680_){
_start:
{
lean_object* v_res_681_; 
v_res_681_ = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_();
return v_res_681_;
}
}
static lean_object* _init_lp_aesop_Aesop_Check_tree(void){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lp_aesop_Aesop_option_00___x40_Aesop_Check_445867947____hygCtx___hyg_3_;
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; 
v___x_707_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_));
v___x_708_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_));
v___x_709_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_));
v___x_710_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(v___x_707_, v___x_708_, v___x_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_611402289____hygCtx___hyg_5____boxed(lean_object* v_a_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_();
return v_res_712_;
}
}
static lean_object* _init_lp_aesop_Aesop_Check_rules(void){
_start:
{
lean_object* v___x_713_; 
v___x_713_ = lp_aesop_Aesop_option_00___x40_Aesop_Check_611402289____hygCtx___hyg_3_;
return v___x_713_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; 
v___x_738_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_));
v___x_739_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_));
v___x_740_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_));
v___x_741_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(v___x_738_, v___x_739_, v___x_740_);
return v___x_741_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5____boxed(lean_object* v_a_742_){
_start:
{
lean_object* v_res_743_; 
v_res_743_ = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_();
return v_res_743_;
}
}
static lean_object* _init_lp_aesop_Aesop_Check_script(void){
_start:
{
lean_object* v___x_744_; 
v___x_744_ = lp_aesop_Aesop_option_00___x40_Aesop_Check_1191392668____hygCtx___hyg_3_;
return v___x_744_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; 
v___x_757_ = lean_unsigned_to_nat(2536445200u);
v___x_758_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__9_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_));
v___x_759_ = l_Lean_Name_num___override(v___x_758_, v___x_757_);
return v___x_759_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; 
v___x_760_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__11_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_));
v___x_761_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_, &lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__once, _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__4_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_);
v___x_762_ = l_Lean_Name_str___override(v___x_761_, v___x_760_);
return v___x_762_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; 
v___x_763_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__13_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_));
v___x_764_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_, &lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__once, _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__5_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_);
v___x_765_ = l_Lean_Name_str___override(v___x_764_, v___x_763_);
return v___x_765_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_766_ = lean_unsigned_to_nat(3u);
v___x_767_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_, &lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__once, _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__6_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_);
v___x_768_ = l_Lean_Name_num___override(v___x_767_, v___x_766_);
return v___x_768_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; 
v___x_770_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__1_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_));
v___x_771_ = ((lean_object*)(lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__3_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_));
v___x_772_ = lean_obj_once(&lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_, &lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5__once, _init_lp_aesop___private_Aesop_Check_0__Aesop_initFn___closed__7_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_);
v___x_773_ = lp_aesop_Lean_Option_register___at___00__private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2__spec__0(v___x_770_, v___x_771_, v___x_772_);
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5____boxed(lean_object* v_a_774_){
_start:
{
lean_object* v_res_775_; 
v_res_775_ = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_();
return v_res_775_;
}
}
static lean_object* _init_lp_aesop_Aesop_Check_script_steps(void){
_start:
{
lean_object* v___x_776_; 
v___x_776_ = lp_aesop_Aesop_option_00___x40_Aesop_Check_2536445200____hygCtx___hyg_3_;
return v___x_776_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_Options(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Check(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2742881586____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_Check_all = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_Check_all);
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_3841271143____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_option_00___x40_Aesop_Check_3841271143____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_option_00___x40_Aesop_Check_3841271143____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_aesop_Aesop_Check_proofReconstruction = _init_lp_aesop_Aesop_Check_proofReconstruction();
lean_mark_persistent(lp_aesop_Aesop_Check_proofReconstruction);
res = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_445867947____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_option_00___x40_Aesop_Check_445867947____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_option_00___x40_Aesop_Check_445867947____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_aesop_Aesop_Check_tree = _init_lp_aesop_Aesop_Check_tree();
lean_mark_persistent(lp_aesop_Aesop_Check_tree);
res = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_611402289____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_option_00___x40_Aesop_Check_611402289____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_option_00___x40_Aesop_Check_611402289____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_aesop_Aesop_Check_rules = _init_lp_aesop_Aesop_Check_rules();
lean_mark_persistent(lp_aesop_Aesop_Check_rules);
res = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_1191392668____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_option_00___x40_Aesop_Check_1191392668____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_option_00___x40_Aesop_Check_1191392668____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_aesop_Aesop_Check_script = _init_lp_aesop_Aesop_Check_script();
lean_mark_persistent(lp_aesop_Aesop_Check_script);
res = lp_aesop___private_Aesop_Check_0__Aesop_initFn_00___x40_Aesop_Check_2536445200____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_option_00___x40_Aesop_Check_2536445200____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_option_00___x40_Aesop_Check_2536445200____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_aesop_Aesop_Check_script_steps = _init_lp_aesop_Aesop_Check_script_steps();
lean_mark_persistent(lp_aesop_Aesop_Check_script_steps);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Check(uint8_t builtin) {
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
lean_object* initialize_Lean_Data_Options(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Check(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Check(builtin);
}
#ifdef __cplusplus
}
#endif
