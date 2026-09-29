// Lean compiler output
// Module: Qq.Typ
// Imports: public import Init public meta import Init public import Lean.Meta.Check
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
uint64_t l_Lean_Expr_hash(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_isLevelDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofLevel(lean_object*);
lean_object* lean_expr_dbg_to_string(lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_instReprExpr_repr(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_unsafeMk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_unsafeMk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_unsafeMk(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_unsafeMk___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_Qq_Qq_instBEqQuoted___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instBEqQuoted___aux__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_Qq_Qq_instBEqQuoted___aux__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instBEqQuoted___aux__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instBEqQuoted(lean_object*);
LEAN_EXPORT uint64_t lp_Qq_Qq_instHashableQuoted___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instHashableQuoted___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT uint64_t lp_Qq_Qq_instHashableQuoted___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instHashableQuoted___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instHashableQuoted(lean_object*);
static const lean_string_object lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "_inhabitedExprDummy"};
static const lean_object* lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__0 = (const lean_object*)&lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 247, 56, 151, 29, 116, 116, 243)}};
static const lean_object* lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__1 = (const lean_object*)&lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__1_value;
static lean_once_cell_t lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2;
LEAN_EXPORT lean_object* lp_Qq_Qq_instInhabitedQuoted___aux__1(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instInhabitedQuoted___aux__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instInhabitedQuoted(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instInhabitedQuoted___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted___aux__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted___aux__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted___aux__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedExpr___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedExpr___lam__0___boxed(lean_object*);
static const lean_closure_object lp_Qq_Qq_instCoeOutQuotedExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_Qq_Qq_instCoeOutQuotedExpr___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Qq_instCoeOutQuotedExpr___closed__0 = (const lean_object*)&lp_Qq_Qq_instCoeOutQuotedExpr___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedExpr(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedExpr___boxed(lean_object*);
static const lean_closure_object lp_Qq_Qq_instCoeOutQuotedMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_MessageData_ofExpr, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_Qq_Qq_instCoeOutQuotedMessageData___closed__0 = (const lean_object*)&lp_Qq_Qq_instCoeOutQuotedMessageData___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedMessageData(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedMessageData___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instToMessageDataQuoted(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_instToMessageDataQuoted___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_ty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_ty___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_ty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_ty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Quoted_check_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Quoted_check_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_Qq_Qq_Quoted_check___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_Qq_Qq_Quoted_check___closed__0 = (const lean_object*)&lp_Qq_Qq_Quoted_check___closed__0_value;
static const lean_string_object lp_Qq_Qq_Quoted_check___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "type mismatch"};
static const lean_object* lp_Qq_Qq_Quoted_check___closed__1 = (const lean_object*)&lp_Qq_Qq_Quoted_check___closed__1_value;
static lean_once_cell_t lp_Qq_Qq_Quoted_check___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Quoted_check___closed__2;
static const lean_string_object lp_Qq_Qq_Quoted_check___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_Qq_Qq_Quoted_check___closed__3 = (const lean_object*)&lp_Qq_Qq_Quoted_check___closed__3_value;
static lean_once_cell_t lp_Qq_Qq_Quoted_check___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Quoted_check___closed__4;
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_check(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_check___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_QuotedDefEq_check___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " and "};
static const lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg___closed__0 = (const lean_object*)&lp_Qq_Qq_QuotedDefEq_check___redArg___closed__0_value;
static lean_once_cell_t lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1;
static const lean_string_object lp_Qq_Qq_QuotedDefEq_check___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = " are not defeq"};
static const lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg___closed__2 = (const lean_object*)&lp_Qq_Qq_QuotedDefEq_check___redArg___closed__2_value;
static lean_once_cell_t lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3;
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedDefEq_check(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedDefEq_check___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedLevelDefEq_check___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedLevelDefEq_check___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedLevelDefEq_check(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedLevelDefEq_check___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_unsafeMk___redArg(lean_object* v_e_1_){
_start:
{
lean_inc_ref(v_e_1_);
return v_e_1_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_unsafeMk___redArg___boxed(lean_object* v_e_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_Qq_Qq_Quoted_unsafeMk___redArg(v_e_2_);
lean_dec_ref(v_e_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_unsafeMk(lean_object* v_00_u03b1_4_, lean_object* v_e_5_){
_start:
{
lean_inc_ref(v_e_5_);
return v_e_5_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_unsafeMk___boxed(lean_object* v_00_u03b1_6_, lean_object* v_e_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_Qq_Qq_Quoted_unsafeMk(v_00_u03b1_6_, v_e_7_);
lean_dec_ref(v_e_7_);
lean_dec_ref(v_00_u03b1_6_);
return v_res_8_;
}
}
LEAN_EXPORT uint8_t lp_Qq_Qq_instBEqQuoted___aux__1___redArg(lean_object* v_a_9_, lean_object* v_b_10_){
_start:
{
uint8_t v___x_11_; 
v___x_11_ = lean_expr_eqv(v_a_9_, v_b_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instBEqQuoted___aux__1___redArg___boxed(lean_object* v_a_12_, lean_object* v_b_13_){
_start:
{
uint8_t v_res_14_; lean_object* v_r_15_; 
v_res_14_ = lp_Qq_Qq_instBEqQuoted___aux__1___redArg(v_a_12_, v_b_13_);
lean_dec_ref(v_b_13_);
lean_dec_ref(v_a_12_);
v_r_15_ = lean_box(v_res_14_);
return v_r_15_;
}
}
LEAN_EXPORT uint8_t lp_Qq_Qq_instBEqQuoted___aux__1(lean_object* v_00_u03b1_16_, lean_object* v_a_17_, lean_object* v_b_18_){
_start:
{
uint8_t v___x_19_; 
v___x_19_ = lean_expr_eqv(v_a_17_, v_b_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instBEqQuoted___aux__1___boxed(lean_object* v_00_u03b1_20_, lean_object* v_a_21_, lean_object* v_b_22_){
_start:
{
uint8_t v_res_23_; lean_object* v_r_24_; 
v_res_23_ = lp_Qq_Qq_instBEqQuoted___aux__1(v_00_u03b1_20_, v_a_21_, v_b_22_);
lean_dec_ref(v_b_22_);
lean_dec_ref(v_a_21_);
lean_dec_ref(v_00_u03b1_20_);
v_r_24_ = lean_box(v_res_23_);
return v_r_24_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instBEqQuoted(lean_object* v_00_u03b1_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lean_alloc_closure((void*)(lp_Qq_Qq_instBEqQuoted___aux__1___boxed), 3, 1);
lean_closure_set(v___x_26_, 0, v_00_u03b1_25_);
return v___x_26_;
}
}
LEAN_EXPORT uint64_t lp_Qq_Qq_instHashableQuoted___aux__1___redArg(lean_object* v_e_27_){
_start:
{
uint64_t v___x_28_; 
v___x_28_ = l_Lean_Expr_hash(v_e_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instHashableQuoted___aux__1___redArg___boxed(lean_object* v_e_29_){
_start:
{
uint64_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_Qq_Qq_instHashableQuoted___aux__1___redArg(v_e_29_);
lean_dec_ref(v_e_29_);
v_r_31_ = lean_box_uint64(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT uint64_t lp_Qq_Qq_instHashableQuoted___aux__1(lean_object* v_00_u03b1_32_, lean_object* v_e_33_){
_start:
{
uint64_t v___x_34_; 
v___x_34_ = l_Lean_Expr_hash(v_e_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instHashableQuoted___aux__1___boxed(lean_object* v_00_u03b1_35_, lean_object* v_e_36_){
_start:
{
uint64_t v_res_37_; lean_object* v_r_38_; 
v_res_37_ = lp_Qq_Qq_instHashableQuoted___aux__1(v_00_u03b1_35_, v_e_36_);
lean_dec_ref(v_e_36_);
lean_dec_ref(v_00_u03b1_35_);
v_r_38_ = lean_box_uint64(v_res_37_);
return v_r_38_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instHashableQuoted(lean_object* v_00_u03b1_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lean_alloc_closure((void*)(lp_Qq_Qq_instHashableQuoted___aux__1___boxed), 2, 1);
lean_closure_set(v___x_40_, 0, v_00_u03b1_39_);
return v___x_40_;
}
}
static lean_object* _init_lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2(void){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_44_ = lean_box(0);
v___x_45_ = ((lean_object*)(lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__1));
v___x_46_ = l_Lean_Expr_const___override(v___x_45_, v___x_44_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instInhabitedQuoted___aux__1(lean_object* v_00_u03b1_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lean_obj_once(&lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2, &lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2_once, _init_lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instInhabitedQuoted___aux__1___boxed(lean_object* v_00_u03b1_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_Qq_Qq_instInhabitedQuoted___aux__1(v_00_u03b1_49_);
lean_dec_ref(v_00_u03b1_49_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instInhabitedQuoted(lean_object* v_00_u03b1_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lean_obj_once(&lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2, &lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2_once, _init_lp_Qq_Qq_instInhabitedQuoted___aux__1___closed__2);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instInhabitedQuoted___boxed(lean_object* v_00_u03b1_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_Qq_Qq_instInhabitedQuoted(v_00_u03b1_53_);
lean_dec_ref(v_00_u03b1_53_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted___aux__1___redArg(lean_object* v_e_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lean_expr_dbg_to_string(v_e_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted___aux__1___redArg___boxed(lean_object* v_e_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_Qq_Qq_instToStringQuoted___aux__1___redArg(v_e_57_);
lean_dec_ref(v_e_57_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted___aux__1(lean_object* v_00_u03b1_59_, lean_object* v_e_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lean_expr_dbg_to_string(v_e_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted___aux__1___boxed(lean_object* v_00_u03b1_62_, lean_object* v_e_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_Qq_Qq_instToStringQuoted___aux__1(v_00_u03b1_62_, v_e_63_);
lean_dec_ref(v_e_63_);
lean_dec_ref(v_00_u03b1_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instToStringQuoted(lean_object* v_00_u03b1_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lean_alloc_closure((void*)(lp_Qq_Qq_instToStringQuoted___aux__1___boxed), 2, 1);
lean_closure_set(v___x_66_, 0, v_00_u03b1_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted___aux__1___redArg(lean_object* v_x_67_, lean_object* v_prec_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = l_Lean_instReprExpr_repr(v_x_67_, v_prec_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted___aux__1___redArg___boxed(lean_object* v_x_70_, lean_object* v_prec_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_Qq_Qq_instReprQuoted___aux__1___redArg(v_x_70_, v_prec_71_);
lean_dec(v_prec_71_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted___aux__1(lean_object* v_00_u03b1_73_, lean_object* v_x_74_, lean_object* v_prec_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = l_Lean_instReprExpr_repr(v_x_74_, v_prec_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted___aux__1___boxed(lean_object* v_00_u03b1_77_, lean_object* v_x_78_, lean_object* v_prec_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_Qq_Qq_instReprQuoted___aux__1(v_00_u03b1_77_, v_x_78_, v_prec_79_);
lean_dec(v_prec_79_);
lean_dec_ref(v_00_u03b1_77_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instReprQuoted(lean_object* v_00_u03b1_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lean_alloc_closure((void*)(lp_Qq_Qq_instReprQuoted___aux__1___boxed), 3, 1);
lean_closure_set(v___x_82_, 0, v_00_u03b1_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedExpr___lam__0(lean_object* v_e_83_){
_start:
{
lean_inc_ref(v_e_83_);
return v_e_83_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedExpr___lam__0___boxed(lean_object* v_e_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_Qq_Qq_instCoeOutQuotedExpr___lam__0(v_e_84_);
lean_dec_ref(v_e_84_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedExpr(lean_object* v_00_u03b1_87_){
_start:
{
lean_object* v___f_88_; 
v___f_88_ = ((lean_object*)(lp_Qq_Qq_instCoeOutQuotedExpr___closed__0));
return v___f_88_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedExpr___boxed(lean_object* v_00_u03b1_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_Qq_Qq_instCoeOutQuotedExpr(v_00_u03b1_89_);
lean_dec_ref(v_00_u03b1_89_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedMessageData(lean_object* v_00_u03b1_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = ((lean_object*)(lp_Qq_Qq_instCoeOutQuotedMessageData___closed__0));
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instCoeOutQuotedMessageData___boxed(lean_object* v_00_u03b1_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_Qq_Qq_instCoeOutQuotedMessageData(v_00_u03b1_94_);
lean_dec_ref(v_00_u03b1_94_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instToMessageDataQuoted(lean_object* v_00_u03b1_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = ((lean_object*)(lp_Qq_Qq_instCoeOutQuotedMessageData___closed__0));
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_instToMessageDataQuoted___boxed(lean_object* v_00_u03b1_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_Qq_Qq_instToMessageDataQuoted(v_00_u03b1_98_);
lean_dec_ref(v_00_u03b1_98_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_ty___redArg(lean_object* v_00_u03b1_100_){
_start:
{
lean_inc_ref(v_00_u03b1_100_);
return v_00_u03b1_100_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_ty___redArg___boxed(lean_object* v_00_u03b1_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_Qq_Qq_Quoted_ty___redArg(v_00_u03b1_101_);
lean_dec_ref(v_00_u03b1_101_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_ty(lean_object* v_00_u03b1_103_, lean_object* v_t_104_){
_start:
{
lean_inc_ref(v_00_u03b1_103_);
return v_00_u03b1_103_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_ty___boxed(lean_object* v_00_u03b1_105_, lean_object* v_t_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_Qq_Qq_Quoted_ty(v_00_u03b1_105_, v_t_106_);
lean_dec_ref(v_t_106_);
lean_dec_ref(v_00_u03b1_105_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Quoted_check_spec__0_spec__0(lean_object* v_msgData_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_){
_start:
{
lean_object* v___x_114_; lean_object* v_env_115_; lean_object* v___x_116_; lean_object* v_mctx_117_; lean_object* v_lctx_118_; lean_object* v_options_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_114_ = lean_st_ref_get(v___y_112_);
v_env_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc_ref(v_env_115_);
lean_dec(v___x_114_);
v___x_116_ = lean_st_ref_get(v___y_110_);
v_mctx_117_ = lean_ctor_get(v___x_116_, 0);
lean_inc_ref(v_mctx_117_);
lean_dec(v___x_116_);
v_lctx_118_ = lean_ctor_get(v___y_109_, 2);
v_options_119_ = lean_ctor_get(v___y_111_, 2);
lean_inc_ref(v_options_119_);
lean_inc_ref(v_lctx_118_);
v___x_120_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_120_, 0, v_env_115_);
lean_ctor_set(v___x_120_, 1, v_mctx_117_);
lean_ctor_set(v___x_120_, 2, v_lctx_118_);
lean_ctor_set(v___x_120_, 3, v_options_119_);
v___x_121_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v_msgData_108_);
v___x_122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Quoted_check_spec__0_spec__0___boxed(lean_object* v_msgData_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Quoted_check_spec__0_spec__0(v_msgData_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
lean_dec(v___y_125_);
lean_dec_ref(v___y_124_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg(lean_object* v_msg_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
lean_object* v_ref_136_; lean_object* v___x_137_; lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_146_; 
v_ref_136_ = lean_ctor_get(v___y_133_, 5);
v___x_137_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_Quoted_check_spec__0_spec__0(v_msg_130_, v___y_131_, v___y_132_, v___y_133_, v___y_134_);
v_a_138_ = lean_ctor_get(v___x_137_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_137_);
if (v_isSharedCheck_146_ == 0)
{
v___x_140_ = v___x_137_;
v_isShared_141_ = v_isSharedCheck_146_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_137_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_146_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_142_; lean_object* v___x_144_; 
lean_inc(v_ref_136_);
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v_ref_136_);
lean_ctor_set(v___x_142_, 1, v_a_138_);
if (v_isShared_141_ == 0)
{
lean_ctor_set_tag(v___x_140_, 1);
lean_ctor_set(v___x_140_, 0, v___x_142_);
v___x_144_ = v___x_140_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v___x_142_);
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
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg___boxed(lean_object* v_msg_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg(v_msg_147_, v___y_148_, v___y_149_, v___y_150_, v___y_151_);
lean_dec(v___y_151_);
lean_dec_ref(v___y_150_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
return v_res_153_;
}
}
static lean_object* _init_lp_Qq_Qq_Quoted_check___closed__2(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_157_ = ((lean_object*)(lp_Qq_Qq_Quoted_check___closed__1));
v___x_158_ = l_Lean_stringToMessageData(v___x_157_);
return v___x_158_;
}
}
static lean_object* _init_lp_Qq_Qq_Quoted_check___closed__4(void){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_160_ = ((lean_object*)(lp_Qq_Qq_Quoted_check___closed__3));
v___x_161_ = l_Lean_stringToMessageData(v___x_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_check(lean_object* v_00_u03b1_162_, lean_object* v_e_163_, lean_object* v_a_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_){
_start:
{
lean_object* v___x_169_; 
lean_inc(v_a_167_);
lean_inc_ref(v_a_166_);
lean_inc(v_a_165_);
lean_inc_ref(v_a_164_);
lean_inc_ref(v_e_163_);
v___x_169_ = lean_infer_type(v_e_163_, v_a_164_, v_a_165_, v_a_166_, v_a_167_);
if (lean_obj_tag(v___x_169_) == 0)
{
lean_object* v_a_170_; lean_object* v___x_171_; 
v_a_170_ = lean_ctor_get(v___x_169_, 0);
lean_inc_n(v_a_170_, 2);
lean_dec_ref_known(v___x_169_, 1);
lean_inc_ref(v_00_u03b1_162_);
v___x_171_ = l_Lean_Meta_isExprDefEq(v_00_u03b1_162_, v_a_170_, v_a_164_, v_a_165_, v_a_166_, v_a_167_);
if (lean_obj_tag(v___x_171_) == 0)
{
lean_object* v_a_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_200_; 
v_a_172_ = lean_ctor_get(v___x_171_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_171_);
if (v_isSharedCheck_200_ == 0)
{
v___x_174_ = v___x_171_;
v_isShared_175_ = v_isSharedCheck_200_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_a_172_);
lean_dec(v___x_171_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_200_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
uint8_t v___x_176_; 
v___x_176_ = lean_unbox(v_a_172_);
lean_dec(v_a_172_);
if (v___x_176_ == 0)
{
lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
lean_del_object(v___x_174_);
v___x_177_ = lean_box(0);
v___x_178_ = ((lean_object*)(lp_Qq_Qq_Quoted_check___closed__0));
v___x_179_ = l_Lean_Meta_mkHasTypeButIsExpectedMsg___redArg(v_a_170_, v_00_u03b1_162_, v___x_177_, v___x_178_);
if (lean_obj_tag(v___x_179_) == 0)
{
lean_object* v_a_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v_a_180_ = lean_ctor_get(v___x_179_, 0);
lean_inc(v_a_180_);
lean_dec_ref_known(v___x_179_, 1);
v___x_181_ = lean_obj_once(&lp_Qq_Qq_Quoted_check___closed__2, &lp_Qq_Qq_Quoted_check___closed__2_once, _init_lp_Qq_Qq_Quoted_check___closed__2);
v___x_182_ = l_Lean_indentExpr(v_e_163_);
v___x_183_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_181_);
lean_ctor_set(v___x_183_, 1, v___x_182_);
v___x_184_ = lean_obj_once(&lp_Qq_Qq_Quoted_check___closed__4, &lp_Qq_Qq_Quoted_check___closed__4_once, _init_lp_Qq_Qq_Quoted_check___closed__4);
v___x_185_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_185_, 0, v___x_183_);
lean_ctor_set(v___x_185_, 1, v___x_184_);
v___x_186_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_185_);
lean_ctor_set(v___x_186_, 1, v_a_180_);
v___x_187_ = lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg(v___x_186_, v_a_164_, v_a_165_, v_a_166_, v_a_167_);
return v___x_187_;
}
else
{
lean_object* v_a_188_; lean_object* v___x_190_; uint8_t v_isShared_191_; uint8_t v_isSharedCheck_195_; 
lean_dec_ref(v_e_163_);
v_a_188_ = lean_ctor_get(v___x_179_, 0);
v_isSharedCheck_195_ = !lean_is_exclusive(v___x_179_);
if (v_isSharedCheck_195_ == 0)
{
v___x_190_ = v___x_179_;
v_isShared_191_ = v_isSharedCheck_195_;
goto v_resetjp_189_;
}
else
{
lean_inc(v_a_188_);
lean_dec(v___x_179_);
v___x_190_ = lean_box(0);
v_isShared_191_ = v_isSharedCheck_195_;
goto v_resetjp_189_;
}
v_resetjp_189_:
{
lean_object* v___x_193_; 
if (v_isShared_191_ == 0)
{
v___x_193_ = v___x_190_;
goto v_reusejp_192_;
}
else
{
lean_object* v_reuseFailAlloc_194_; 
v_reuseFailAlloc_194_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_194_, 0, v_a_188_);
v___x_193_ = v_reuseFailAlloc_194_;
goto v_reusejp_192_;
}
v_reusejp_192_:
{
return v___x_193_;
}
}
}
}
else
{
lean_object* v___x_196_; lean_object* v___x_198_; 
lean_dec(v_a_170_);
lean_dec_ref(v_e_163_);
lean_dec_ref(v_00_u03b1_162_);
v___x_196_ = lean_box(0);
if (v_isShared_175_ == 0)
{
lean_ctor_set(v___x_174_, 0, v___x_196_);
v___x_198_ = v___x_174_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v___x_196_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
}
else
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_208_; 
lean_dec(v_a_170_);
lean_dec_ref(v_e_163_);
lean_dec_ref(v_00_u03b1_162_);
v_a_201_ = lean_ctor_get(v___x_171_, 0);
v_isSharedCheck_208_ = !lean_is_exclusive(v___x_171_);
if (v_isSharedCheck_208_ == 0)
{
v___x_203_ = v___x_171_;
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_171_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_208_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v___x_206_; 
if (v_isShared_204_ == 0)
{
v___x_206_ = v___x_203_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v_a_201_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
else
{
lean_object* v_a_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_216_; 
lean_dec_ref(v_e_163_);
lean_dec_ref(v_00_u03b1_162_);
v_a_209_ = lean_ctor_get(v___x_169_, 0);
v_isSharedCheck_216_ = !lean_is_exclusive(v___x_169_);
if (v_isSharedCheck_216_ == 0)
{
v___x_211_ = v___x_169_;
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_a_209_);
lean_dec(v___x_169_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_214_; 
if (v_isShared_212_ == 0)
{
v___x_214_ = v___x_211_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v_a_209_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Quoted_check___boxed(lean_object* v_00_u03b1_217_, lean_object* v_e_218_, lean_object* v_a_219_, lean_object* v_a_220_, lean_object* v_a_221_, lean_object* v_a_222_, lean_object* v_a_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_Qq_Qq_Quoted_check(v_00_u03b1_217_, v_e_218_, v_a_219_, v_a_220_, v_a_221_, v_a_222_);
lean_dec(v_a_222_);
lean_dec_ref(v_a_221_);
lean_dec(v_a_220_);
lean_dec_ref(v_a_219_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0(lean_object* v_00_u03b1_225_, lean_object* v_msg_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg(v_msg_226_, v___y_227_, v___y_228_, v___y_229_, v___y_230_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___boxed(lean_object* v_00_u03b1_233_, lean_object* v_msg_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0(v_00_u03b1_233_, v_msg_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_);
lean_dec(v___y_238_);
lean_dec_ref(v___y_237_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
return v_res_240_;
}
}
static lean_object* _init_lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1(void){
_start:
{
lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_242_ = ((lean_object*)(lp_Qq_Qq_QuotedDefEq_check___redArg___closed__0));
v___x_243_ = l_Lean_stringToMessageData(v___x_242_);
return v___x_243_;
}
}
static lean_object* _init_lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3(void){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_245_ = ((lean_object*)(lp_Qq_Qq_QuotedDefEq_check___redArg___closed__2));
v___x_246_ = l_Lean_stringToMessageData(v___x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg(lean_object* v_u_247_, lean_object* v_00_u03b1_248_, lean_object* v_lhs_249_, lean_object* v_rhs_250_, lean_object* v_a_251_, lean_object* v_a_252_, lean_object* v_a_253_, lean_object* v_a_254_){
_start:
{
lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_256_ = l_Lean_Expr_sort___override(v_u_247_);
lean_inc_ref(v_00_u03b1_248_);
v___x_257_ = lp_Qq_Qq_Quoted_check(v___x_256_, v_00_u03b1_248_, v_a_251_, v_a_252_, v_a_253_, v_a_254_);
if (lean_obj_tag(v___x_257_) == 0)
{
lean_object* v___x_258_; 
lean_dec_ref_known(v___x_257_, 1);
lean_inc_ref(v_lhs_249_);
lean_inc_ref(v_00_u03b1_248_);
v___x_258_ = lp_Qq_Qq_Quoted_check(v_00_u03b1_248_, v_lhs_249_, v_a_251_, v_a_252_, v_a_253_, v_a_254_);
if (lean_obj_tag(v___x_258_) == 0)
{
lean_object* v___x_259_; 
lean_dec_ref_known(v___x_258_, 1);
lean_inc_ref(v_rhs_250_);
v___x_259_ = lp_Qq_Qq_Quoted_check(v_00_u03b1_248_, v_rhs_250_, v_a_251_, v_a_252_, v_a_253_, v_a_254_);
if (lean_obj_tag(v___x_259_) == 0)
{
lean_object* v___x_260_; 
lean_dec_ref_known(v___x_259_, 1);
lean_inc_ref(v_rhs_250_);
lean_inc_ref(v_lhs_249_);
v___x_260_ = l_Lean_Meta_isExprDefEq(v_lhs_249_, v_rhs_250_, v_a_251_, v_a_252_, v_a_253_, v_a_254_);
if (lean_obj_tag(v___x_260_) == 0)
{
lean_object* v_a_261_; lean_object* v___x_263_; uint8_t v_isShared_264_; uint8_t v_isSharedCheck_278_; 
v_a_261_ = lean_ctor_get(v___x_260_, 0);
v_isSharedCheck_278_ = !lean_is_exclusive(v___x_260_);
if (v_isSharedCheck_278_ == 0)
{
v___x_263_ = v___x_260_;
v_isShared_264_ = v_isSharedCheck_278_;
goto v_resetjp_262_;
}
else
{
lean_inc(v_a_261_);
lean_dec(v___x_260_);
v___x_263_ = lean_box(0);
v_isShared_264_ = v_isSharedCheck_278_;
goto v_resetjp_262_;
}
v_resetjp_262_:
{
uint8_t v___x_265_; 
v___x_265_ = lean_unbox(v_a_261_);
lean_dec(v_a_261_);
if (v___x_265_ == 0)
{
lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; 
lean_del_object(v___x_263_);
v___x_266_ = l_Lean_MessageData_ofExpr(v_lhs_249_);
v___x_267_ = lean_obj_once(&lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1, &lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1_once, _init_lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1);
v___x_268_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_268_, 0, v___x_266_);
lean_ctor_set(v___x_268_, 1, v___x_267_);
v___x_269_ = l_Lean_MessageData_ofExpr(v_rhs_250_);
v___x_270_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_270_, 0, v___x_268_);
lean_ctor_set(v___x_270_, 1, v___x_269_);
v___x_271_ = lean_obj_once(&lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3, &lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3_once, _init_lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3);
v___x_272_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_270_);
lean_ctor_set(v___x_272_, 1, v___x_271_);
v___x_273_ = lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg(v___x_272_, v_a_251_, v_a_252_, v_a_253_, v_a_254_);
return v___x_273_;
}
else
{
lean_object* v___x_274_; lean_object* v___x_276_; 
lean_dec_ref(v_rhs_250_);
lean_dec_ref(v_lhs_249_);
v___x_274_ = lean_box(0);
if (v_isShared_264_ == 0)
{
lean_ctor_set(v___x_263_, 0, v___x_274_);
v___x_276_ = v___x_263_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v___x_274_);
v___x_276_ = v_reuseFailAlloc_277_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
return v___x_276_;
}
}
}
}
else
{
lean_object* v_a_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_286_; 
lean_dec_ref(v_rhs_250_);
lean_dec_ref(v_lhs_249_);
v_a_279_ = lean_ctor_get(v___x_260_, 0);
v_isSharedCheck_286_ = !lean_is_exclusive(v___x_260_);
if (v_isSharedCheck_286_ == 0)
{
v___x_281_ = v___x_260_;
v_isShared_282_ = v_isSharedCheck_286_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_a_279_);
lean_dec(v___x_260_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_286_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___x_284_; 
if (v_isShared_282_ == 0)
{
v___x_284_ = v___x_281_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_a_279_);
v___x_284_ = v_reuseFailAlloc_285_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
return v___x_284_;
}
}
}
}
else
{
lean_dec_ref(v_rhs_250_);
lean_dec_ref(v_lhs_249_);
return v___x_259_;
}
}
else
{
lean_dec_ref(v_rhs_250_);
lean_dec_ref(v_lhs_249_);
lean_dec_ref(v_00_u03b1_248_);
return v___x_258_;
}
}
else
{
lean_dec_ref(v_rhs_250_);
lean_dec_ref(v_lhs_249_);
lean_dec_ref(v_00_u03b1_248_);
return v___x_257_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedDefEq_check___redArg___boxed(lean_object* v_u_287_, lean_object* v_00_u03b1_288_, lean_object* v_lhs_289_, lean_object* v_rhs_290_, lean_object* v_a_291_, lean_object* v_a_292_, lean_object* v_a_293_, lean_object* v_a_294_, lean_object* v_a_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_Qq_Qq_QuotedDefEq_check___redArg(v_u_287_, v_00_u03b1_288_, v_lhs_289_, v_rhs_290_, v_a_291_, v_a_292_, v_a_293_, v_a_294_);
lean_dec(v_a_294_);
lean_dec_ref(v_a_293_);
lean_dec(v_a_292_);
lean_dec_ref(v_a_291_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedDefEq_check(lean_object* v_u_297_, lean_object* v_00_u03b1_298_, lean_object* v_lhs_299_, lean_object* v_rhs_300_, lean_object* v_e_301_, lean_object* v_a_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_Qq_Qq_QuotedDefEq_check___redArg(v_u_297_, v_00_u03b1_298_, v_lhs_299_, v_rhs_300_, v_a_302_, v_a_303_, v_a_304_, v_a_305_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedDefEq_check___boxed(lean_object* v_u_308_, lean_object* v_00_u03b1_309_, lean_object* v_lhs_310_, lean_object* v_rhs_311_, lean_object* v_e_312_, lean_object* v_a_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_, lean_object* v_a_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_Qq_Qq_QuotedDefEq_check(v_u_308_, v_00_u03b1_309_, v_lhs_310_, v_rhs_311_, v_e_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
lean_dec(v_a_316_);
lean_dec_ref(v_a_315_);
lean_dec(v_a_314_);
lean_dec_ref(v_a_313_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedLevelDefEq_check___redArg(lean_object* v_lhs_319_, lean_object* v_rhs_320_, lean_object* v_a_321_, lean_object* v_a_322_, lean_object* v_a_323_, lean_object* v_a_324_){
_start:
{
lean_object* v___x_326_; 
lean_inc(v_rhs_320_);
lean_inc(v_lhs_319_);
v___x_326_ = l_Lean_Meta_isLevelDefEq(v_lhs_319_, v_rhs_320_, v_a_321_, v_a_322_, v_a_323_, v_a_324_);
if (lean_obj_tag(v___x_326_) == 0)
{
lean_object* v_a_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_344_; 
v_a_327_ = lean_ctor_get(v___x_326_, 0);
v_isSharedCheck_344_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_344_ == 0)
{
v___x_329_ = v___x_326_;
v_isShared_330_ = v_isSharedCheck_344_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_a_327_);
lean_dec(v___x_326_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_344_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
uint8_t v___x_331_; 
v___x_331_ = lean_unbox(v_a_327_);
lean_dec(v_a_327_);
if (v___x_331_ == 0)
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
lean_del_object(v___x_329_);
v___x_332_ = l_Lean_MessageData_ofLevel(v_lhs_319_);
v___x_333_ = lean_obj_once(&lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1, &lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1_once, _init_lp_Qq_Qq_QuotedDefEq_check___redArg___closed__1);
v___x_334_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_334_, 0, v___x_332_);
lean_ctor_set(v___x_334_, 1, v___x_333_);
v___x_335_ = l_Lean_MessageData_ofLevel(v_rhs_320_);
v___x_336_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_334_);
lean_ctor_set(v___x_336_, 1, v___x_335_);
v___x_337_ = lean_obj_once(&lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3, &lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3_once, _init_lp_Qq_Qq_QuotedDefEq_check___redArg___closed__3);
v___x_338_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_336_);
lean_ctor_set(v___x_338_, 1, v___x_337_);
v___x_339_ = lp_Qq_Lean_throwError___at___00Qq_Quoted_check_spec__0___redArg(v___x_338_, v_a_321_, v_a_322_, v_a_323_, v_a_324_);
return v___x_339_;
}
else
{
lean_object* v___x_340_; lean_object* v___x_342_; 
lean_dec(v_rhs_320_);
lean_dec(v_lhs_319_);
v___x_340_ = lean_box(0);
if (v_isShared_330_ == 0)
{
lean_ctor_set(v___x_329_, 0, v___x_340_);
v___x_342_ = v___x_329_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v___x_340_);
v___x_342_ = v_reuseFailAlloc_343_;
goto v_reusejp_341_;
}
v_reusejp_341_:
{
return v___x_342_;
}
}
}
}
else
{
lean_object* v_a_345_; lean_object* v___x_347_; uint8_t v_isShared_348_; uint8_t v_isSharedCheck_352_; 
lean_dec(v_rhs_320_);
lean_dec(v_lhs_319_);
v_a_345_ = lean_ctor_get(v___x_326_, 0);
v_isSharedCheck_352_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_352_ == 0)
{
v___x_347_ = v___x_326_;
v_isShared_348_ = v_isSharedCheck_352_;
goto v_resetjp_346_;
}
else
{
lean_inc(v_a_345_);
lean_dec(v___x_326_);
v___x_347_ = lean_box(0);
v_isShared_348_ = v_isSharedCheck_352_;
goto v_resetjp_346_;
}
v_resetjp_346_:
{
lean_object* v___x_350_; 
if (v_isShared_348_ == 0)
{
v___x_350_ = v___x_347_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_351_; 
v_reuseFailAlloc_351_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_351_, 0, v_a_345_);
v___x_350_ = v_reuseFailAlloc_351_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
return v___x_350_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedLevelDefEq_check___redArg___boxed(lean_object* v_lhs_353_, lean_object* v_rhs_354_, lean_object* v_a_355_, lean_object* v_a_356_, lean_object* v_a_357_, lean_object* v_a_358_, lean_object* v_a_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_Qq_Qq_QuotedLevelDefEq_check___redArg(v_lhs_353_, v_rhs_354_, v_a_355_, v_a_356_, v_a_357_, v_a_358_);
lean_dec(v_a_358_);
lean_dec_ref(v_a_357_);
lean_dec(v_a_356_);
lean_dec_ref(v_a_355_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedLevelDefEq_check(lean_object* v_lhs_361_, lean_object* v_rhs_362_, lean_object* v_e_363_, lean_object* v_a_364_, lean_object* v_a_365_, lean_object* v_a_366_, lean_object* v_a_367_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lp_Qq_Qq_QuotedLevelDefEq_check___redArg(v_lhs_361_, v_rhs_362_, v_a_364_, v_a_365_, v_a_366_, v_a_367_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_QuotedLevelDefEq_check___boxed(lean_object* v_lhs_370_, lean_object* v_rhs_371_, lean_object* v_e_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_){
_start:
{
lean_object* v_res_378_; 
v_res_378_ = lp_Qq_Qq_QuotedLevelDefEq_check(v_lhs_370_, v_rhs_371_, v_e_372_, v_a_373_, v_a_374_, v_a_375_, v_a_376_);
lean_dec(v_a_376_);
lean_dec_ref(v_a_375_);
lean_dec(v_a_374_);
lean_dec_ref(v_a_373_);
return v_res_378_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Check(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_Typ(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_Typ(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Check(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_Typ(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Check(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_Typ(builtin);
}
#ifdef __cplusplus
}
#endif
