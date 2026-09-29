// Lean compiler output
// Module: Batteries.Lean.Position
// Imports: public import Init public meta import Init public import Lean.Syntax public import Lean.Data.Lsp.Utf16
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_String_Slice_posLE(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* l_Lean_FileMap_ofPosition(lean_object*, lean_object*);
uint8_t l_Lean_Environment_isImportedConst(lean_object*, lean_object*);
lean_object* l_Lean_findDeclarationRanges_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_Pos_next_x21(lean_object*, lean_object*);
lean_object* l_String_Slice_pos_x21(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_ofRange(lean_object*, uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_declRangeExt;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Position_lt(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findLineStart(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findLineStart___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findIndentAndIsStart(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findIndentAndIsStart___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Position_getDeclsAfter_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_batteries_Lean_Position_getDeclsAfter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_Position_getDeclsAfter___closed__0 = (const lean_object*)&lp_batteries_Lean_Position_getDeclsAfter___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Position_getDeclsAfter(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Position_getDeclsAfter___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Position_getDeclsAfter_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Pos_Raw_getDeclsAfter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_String_Pos_Raw_getDeclsAfter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_DeclarationRange_toSyntaxRange(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_DeclarationRange_toSyntaxRange___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(lean_object* v_s_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
v___x_4_ = lean_unsigned_to_nat(0u);
v___x_5_ = lean_nat_dec_eq(v_a_2_, v___x_4_);
if (v___x_5_ == 0)
{
lean_object* v_str_6_; lean_object* v_startInclusive_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; uint32_t v___x_15_; uint32_t v___x_16_; uint8_t v___x_17_; 
v_str_6_ = lean_ctor_get(v_s_1_, 0);
v_startInclusive_7_ = lean_ctor_get(v_s_1_, 1);
v___x_8_ = lean_nat_add(v_startInclusive_7_, v_a_2_);
lean_inc(v___x_8_);
lean_inc(v_startInclusive_7_);
lean_inc_ref(v_str_6_);
v___x_9_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_9_, 0, v_str_6_);
lean_ctor_set(v___x_9_, 1, v_startInclusive_7_);
lean_ctor_set(v___x_9_, 2, v___x_8_);
v___x_10_ = lean_nat_sub(v___x_8_, v_startInclusive_7_);
lean_dec(v___x_8_);
v___x_11_ = lean_unsigned_to_nat(1u);
v___x_12_ = lean_nat_sub(v___x_10_, v___x_11_);
lean_dec(v___x_10_);
v___x_13_ = l_String_Slice_posLE(v___x_9_, v___x_12_);
lean_dec_ref_known(v___x_9_, 3);
v___x_14_ = lean_nat_add(v_startInclusive_7_, v___x_13_);
v___x_15_ = lean_string_utf8_get_fast(v_str_6_, v___x_14_);
lean_dec(v___x_14_);
v___x_16_ = 10;
v___x_17_ = lean_uint32_dec_eq(v___x_15_, v___x_16_);
if (v___x_17_ == 0)
{
lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
lean_dec(v___x_13_);
v___x_18_ = lean_box(0);
v___x_19_ = lean_nat_sub(v_a_2_, v___x_11_);
lean_dec(v_a_2_);
v___x_20_ = l_String_Slice_posLE(v_s_1_, v___x_19_);
v_a_2_ = v___x_20_;
v_b_3_ = v___x_18_;
goto _start;
}
else
{
lean_object* v___x_22_; 
lean_dec(v_a_2_);
v___x_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_22_, 0, v___x_13_);
return v___x_22_;
}
}
else
{
lean_dec(v_a_2_);
lean_inc(v_b_3_);
return v_b_3_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg___boxed(lean_object* v_s_23_, lean_object* v_a_24_, lean_object* v_b_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(v_s_23_, v_a_24_, v_b_25_);
lean_dec(v_b_25_);
lean_dec_ref(v_s_23_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0(lean_object* v_s_27_){
_start:
{
lean_object* v_startInclusive_28_; lean_object* v_endExclusive_29_; lean_object* v_searcher_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v_startInclusive_28_ = lean_ctor_get(v_s_27_, 1);
v_endExclusive_29_ = lean_ctor_get(v_s_27_, 2);
v_searcher_30_ = lean_nat_sub(v_endExclusive_29_, v_startInclusive_28_);
v___x_31_ = lean_box(0);
v___x_32_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(v_s_27_, v_searcher_30_, v___x_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0___boxed(lean_object* v_s_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0(v_s_33_);
lean_dec_ref(v_s_33_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findLineStart(lean_object* v_s_35_, lean_object* v_pos_36_){
_start:
{
lean_object* v_val_38_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_43_ = lean_unsigned_to_nat(0u);
v___x_44_ = lean_string_utf8_byte_size(v_s_35_);
lean_inc_ref_n(v_s_35_, 2);
v___x_45_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_45_, 0, v_s_35_);
lean_ctor_set(v___x_45_, 1, v___x_43_);
lean_ctor_set(v___x_45_, 2, v___x_44_);
v___x_46_ = l_String_Slice_pos_x21(v___x_45_, v_pos_36_);
lean_dec_ref_known(v___x_45_, 3);
v___x_47_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_47_, 0, v_s_35_);
lean_ctor_set(v___x_47_, 1, v___x_43_);
lean_ctor_set(v___x_47_, 2, v___x_46_);
v___x_48_ = lp_batteries_String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0(v___x_47_);
lean_dec_ref_known(v___x_47_, 3);
if (lean_obj_tag(v___x_48_) == 0)
{
if (lean_obj_tag(v___x_48_) == 0)
{
lean_dec_ref(v_s_35_);
return v___x_43_;
}
else
{
lean_object* v_val_49_; 
v_val_49_ = lean_ctor_get(v___x_48_, 0);
lean_inc(v_val_49_);
lean_dec_ref_known(v___x_48_, 1);
v_val_38_ = v_val_49_;
goto v___jp_37_;
}
}
else
{
lean_object* v_val_50_; 
v_val_50_ = lean_ctor_get(v___x_48_, 0);
lean_inc(v_val_50_);
lean_dec_ref_known(v___x_48_, 1);
v_val_38_ = v_val_50_;
goto v___jp_37_;
}
v___jp_37_:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_39_ = lean_unsigned_to_nat(0u);
v___x_40_ = lean_string_utf8_byte_size(v_s_35_);
v___x_41_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_41_, 0, v_s_35_);
lean_ctor_set(v___x_41_, 1, v___x_39_);
lean_ctor_set(v___x_41_, 2, v___x_40_);
v___x_42_ = l_String_Slice_Pos_next_x21(v___x_41_, v_val_38_);
lean_dec(v_val_38_);
lean_dec_ref_known(v___x_41_, 3);
return v___x_42_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findLineStart___boxed(lean_object* v_s_51_, lean_object* v_pos_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_batteries_Lean_findLineStart(v_s_51_, v_pos_52_);
lean_dec(v_pos_52_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0(lean_object* v_s_54_, lean_object* v_inst_55_, lean_object* v_R_56_, lean_object* v_a_57_, lean_object* v_b_58_, lean_object* v_c_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___redArg(v_s_54_, v_a_57_, v_b_58_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0___boxed(lean_object* v_s_61_, lean_object* v_inst_62_, lean_object* v_R_63_, lean_object* v_a_64_, lean_object* v_b_65_, lean_object* v_c_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00String_Slice_revFind_x3f___at___00Lean_findLineStart_spec__0_spec__0(v_s_61_, v_inst_62_, v_R_63_, v_a_64_, v_b_65_, v_c_66_);
lean_dec(v_b_65_);
lean_dec_ref(v_s_61_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(lean_object* v___x_68_, lean_object* v___x_69_, lean_object* v_s_70_, lean_object* v_a_71_, lean_object* v_b_72_){
_start:
{
lean_object* v_startInclusive_73_; lean_object* v_endExclusive_74_; lean_object* v___x_75_; uint8_t v___x_76_; 
v_startInclusive_73_ = lean_ctor_get(v___x_68_, 1);
v_endExclusive_74_ = lean_ctor_get(v___x_68_, 2);
v___x_75_ = lean_nat_sub(v_endExclusive_74_, v_startInclusive_73_);
v___x_76_ = lean_nat_dec_eq(v_a_71_, v___x_75_);
lean_dec(v___x_75_);
if (v___x_76_ == 0)
{
lean_object* v___x_77_; uint32_t v___x_78_; uint32_t v___x_79_; uint8_t v___x_80_; 
v___x_77_ = lean_nat_add(v___x_69_, v_a_71_);
v___x_78_ = lean_string_utf8_get_fast(v_s_70_, v___x_77_);
v___x_79_ = 32;
v___x_80_ = lean_uint32_dec_eq(v___x_78_, v___x_79_);
if (v___x_80_ == 0)
{
lean_object* v___x_81_; 
lean_dec(v___x_77_);
v___x_81_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_81_, 0, v_a_71_);
return v___x_81_;
}
else
{
if (v___x_76_ == 0)
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
lean_dec(v_a_71_);
v___x_82_ = lean_box(0);
v___x_83_ = lean_string_utf8_next_fast(v_s_70_, v___x_77_);
lean_dec(v___x_77_);
v___x_84_ = lean_nat_sub(v___x_83_, v___x_69_);
v_a_71_ = v___x_84_;
v_b_72_ = v___x_82_;
goto _start;
}
else
{
lean_object* v___x_86_; 
lean_dec(v___x_77_);
v___x_86_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_86_, 0, v_a_71_);
return v___x_86_;
}
}
}
else
{
lean_dec(v_a_71_);
lean_inc(v_b_72_);
return v_b_72_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg___boxed(lean_object* v___x_87_, lean_object* v___x_88_, lean_object* v_s_89_, lean_object* v_a_90_, lean_object* v_b_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(v___x_87_, v___x_88_, v_s_89_, v_a_90_, v_b_91_);
lean_dec(v_b_91_);
lean_dec_ref(v_s_89_);
lean_dec(v___x_88_);
lean_dec_ref(v___x_87_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findIndentAndIsStart(lean_object* v_s_93_, lean_object* v_pos_94_){
_start:
{
lean_object* v_start_95_; lean_object* v_searcher_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___y_101_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
lean_inc_ref_n(v_s_93_, 3);
v_start_95_ = lp_batteries_Lean_findLineStart(v_s_93_, v_pos_94_);
v_searcher_96_ = lean_unsigned_to_nat(0u);
v___x_97_ = lean_string_utf8_byte_size(v_s_93_);
v___x_98_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_98_, 0, v_s_93_);
lean_ctor_set(v___x_98_, 1, v_searcher_96_);
lean_ctor_set(v___x_98_, 2, v___x_97_);
v___x_99_ = l_String_Slice_pos_x21(v___x_98_, v_start_95_);
lean_dec_ref_known(v___x_98_, 3);
lean_inc(v___x_99_);
v___x_107_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_107_, 0, v_s_93_);
lean_ctor_set(v___x_107_, 1, v___x_99_);
lean_ctor_set(v___x_107_, 2, v___x_97_);
v___x_108_ = lean_box(0);
v___x_109_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(v___x_107_, v___x_99_, v_s_93_, v_searcher_96_, v___x_108_);
lean_dec_ref(v_s_93_);
lean_dec_ref_known(v___x_107_, 3);
if (lean_obj_tag(v___x_109_) == 0)
{
lean_object* v___x_110_; 
v___x_110_ = lean_nat_sub(v___x_97_, v___x_99_);
v___y_101_ = v___x_110_;
goto v___jp_100_;
}
else
{
lean_object* v_val_111_; 
v_val_111_ = lean_ctor_get(v___x_109_, 0);
lean_inc(v_val_111_);
lean_dec_ref_known(v___x_109_, 1);
v___y_101_ = v_val_111_;
goto v___jp_100_;
}
v___jp_100_:
{
lean_object* v_body_102_; lean_object* v___x_103_; uint8_t v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v_body_102_ = lean_nat_add(v___x_99_, v___y_101_);
lean_dec(v___y_101_);
lean_dec(v___x_99_);
v___x_103_ = lean_nat_sub(v_body_102_, v_start_95_);
lean_dec(v_start_95_);
v___x_104_ = lean_nat_dec_eq(v_body_102_, v_pos_94_);
lean_dec(v_body_102_);
v___x_105_ = lean_box(v___x_104_);
v___x_106_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_106_, 0, v___x_103_);
lean_ctor_set(v___x_106_, 1, v___x_105_);
return v___x_106_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findIndentAndIsStart___boxed(lean_object* v_s_112_, lean_object* v_pos_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_batteries_Lean_findIndentAndIsStart(v_s_112_, v_pos_113_);
lean_dec(v_pos_113_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0(lean_object* v___x_115_, lean_object* v___x_116_, lean_object* v_s_117_, lean_object* v_inst_118_, lean_object* v_R_119_, lean_object* v_a_120_, lean_object* v_b_121_, lean_object* v_c_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___redArg(v___x_115_, v___x_116_, v_s_117_, v_a_120_, v_b_121_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0___boxed(lean_object* v___x_124_, lean_object* v___x_125_, lean_object* v_s_126_, lean_object* v_inst_127_, lean_object* v_R_128_, lean_object* v_a_129_, lean_object* v_b_130_, lean_object* v_c_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_batteries_WellFounded_opaqueFix_u2083___at___00Lean_findIndentAndIsStart_spec__0(v___x_124_, v___x_125_, v_s_126_, v_inst_127_, v_R_128_, v_a_129_, v_b_130_, v_c_131_);
lean_dec(v_b_130_);
lean_dec_ref(v_s_126_);
lean_dec(v___x_125_);
lean_dec_ref(v___x_124_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Position_getDeclsAfter_spec__0_spec__0(lean_object* v_pos_133_, lean_object* v_init_134_, lean_object* v_x_135_){
_start:
{
if (lean_obj_tag(v_x_135_) == 0)
{
lean_object* v_v_136_; lean_object* v_selectionRange_137_; lean_object* v_k_138_; lean_object* v_l_139_; lean_object* v_r_140_; lean_object* v_pos_141_; lean_object* v___x_142_; uint8_t v___x_143_; 
v_v_136_ = lean_ctor_get(v_x_135_, 2);
v_selectionRange_137_ = lean_ctor_get(v_v_136_, 1);
lean_inc_ref(v_selectionRange_137_);
v_k_138_ = lean_ctor_get(v_x_135_, 1);
lean_inc(v_k_138_);
v_l_139_ = lean_ctor_get(v_x_135_, 3);
lean_inc(v_l_139_);
v_r_140_ = lean_ctor_get(v_x_135_, 4);
lean_inc(v_r_140_);
lean_dec_ref_known(v_x_135_, 5);
v_pos_141_ = lean_ctor_get(v_selectionRange_137_, 0);
lean_inc_ref(v_pos_141_);
lean_dec_ref(v_selectionRange_137_);
lean_inc_ref_n(v_pos_133_, 2);
v___x_142_ = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Position_getDeclsAfter_spec__0_spec__0(v_pos_133_, v_init_134_, v_l_139_);
v___x_143_ = l_Lean_Position_lt(v_pos_141_, v_pos_133_);
if (v___x_143_ == 0)
{
lean_object* v___x_144_; 
v___x_144_ = lean_array_push(v___x_142_, v_k_138_);
v_init_134_ = v___x_144_;
v_x_135_ = v_r_140_;
goto _start;
}
else
{
lean_dec(v_k_138_);
v_init_134_ = v___x_142_;
v_x_135_ = v_r_140_;
goto _start;
}
}
else
{
lean_dec_ref(v_pos_133_);
return v_init_134_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Position_getDeclsAfter(lean_object* v_env_149_, lean_object* v_pos_150_, lean_object* v_asyncMode_151_){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_152_ = lean_box(1);
v___x_153_ = ((lean_object*)(lp_batteries_Lean_Position_getDeclsAfter___closed__0));
v___x_154_ = l_Lean_declRangeExt;
v___x_155_ = lean_box(0);
v___x_156_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_152_, v___x_154_, v_env_149_, v_asyncMode_151_, v___x_155_);
v___x_157_ = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Position_getDeclsAfter_spec__0_spec__0(v_pos_150_, v___x_153_, v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Position_getDeclsAfter___boxed(lean_object* v_env_158_, lean_object* v_pos_159_, lean_object* v_asyncMode_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_batteries_Lean_Position_getDeclsAfter(v_env_158_, v_pos_159_, v_asyncMode_160_);
lean_dec(v_asyncMode_160_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Position_getDeclsAfter_spec__0(lean_object* v_pos_162_, lean_object* v_init_163_, lean_object* v_t_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Position_getDeclsAfter_spec__0_spec__0(v_pos_162_, v_init_163_, v_t_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Pos_Raw_getDeclsAfter(lean_object* v_env_166_, lean_object* v_map_167_, lean_object* v_pos_168_, lean_object* v_asyncMode_169_){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_170_ = l_Lean_FileMap_toPosition(v_map_167_, v_pos_168_);
v___x_171_ = lp_batteries_Lean_Position_getDeclsAfter(v_env_166_, v___x_170_, v_asyncMode_169_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_batteries_String_Pos_Raw_getDeclsAfter___boxed(lean_object* v_env_172_, lean_object* v_map_173_, lean_object* v_pos_174_, lean_object* v_asyncMode_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_batteries_String_Pos_Raw_getDeclsAfter(v_env_172_, v_map_173_, v_pos_174_, v_asyncMode_175_);
lean_dec(v_asyncMode_175_);
lean_dec(v_pos_174_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_DeclarationRange_toSyntaxRange(lean_object* v_map_177_, lean_object* v_range_178_){
_start:
{
lean_object* v_pos_179_; lean_object* v_endPos_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; 
v_pos_179_ = lean_ctor_get(v_range_178_, 0);
lean_inc_ref(v_pos_179_);
v_endPos_180_ = lean_ctor_get(v_range_178_, 2);
lean_inc_ref(v_endPos_180_);
lean_dec_ref(v_range_178_);
v___x_181_ = l_Lean_FileMap_ofPosition(v_map_177_, v_pos_179_);
v___x_182_ = l_Lean_FileMap_ofPosition(v_map_177_, v_endPos_180_);
v___x_183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_181_);
lean_ctor_set(v___x_183_, 1, v___x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_DeclarationRange_toSyntaxRange___boxed(lean_object* v_map_184_, lean_object* v_range_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_batteries_Lean_DeclarationRange_toSyntaxRange(v_map_184_, v_range_185_);
lean_dec_ref(v_map_184_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__0(lean_object* v_toApplicative_187_, uint8_t v_fullRange_188_, lean_object* v_val_189_, lean_object* v_____do__lift_190_){
_start:
{
lean_object* v_toPure_191_; lean_object* v___y_193_; 
v_toPure_191_ = lean_ctor_get(v_toApplicative_187_, 1);
lean_inc(v_toPure_191_);
lean_dec_ref(v_toApplicative_187_);
if (v_fullRange_188_ == 0)
{
lean_object* v_selectionRange_197_; 
v_selectionRange_197_ = lean_ctor_get(v_val_189_, 1);
lean_inc_ref(v_selectionRange_197_);
lean_dec_ref(v_val_189_);
v___y_193_ = v_selectionRange_197_;
goto v___jp_192_;
}
else
{
lean_object* v_range_198_; 
v_range_198_ = lean_ctor_get(v_val_189_, 0);
lean_inc_ref(v_range_198_);
lean_dec_ref(v_val_189_);
v___y_193_ = v_range_198_;
goto v___jp_192_;
}
v___jp_192_:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
v___x_194_ = lp_batteries_Lean_DeclarationRange_toSyntaxRange(v_____do__lift_190_, v___y_193_);
v___x_195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_195_, 0, v___x_194_);
v___x_196_ = lean_apply_2(v_toPure_191_, lean_box(0), v___x_195_);
return v___x_196_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__0___boxed(lean_object* v_toApplicative_199_, lean_object* v_fullRange_200_, lean_object* v_val_201_, lean_object* v_____do__lift_202_){
_start:
{
uint8_t v_fullRange_boxed_203_; lean_object* v_res_204_; 
v_fullRange_boxed_203_ = lean_unbox(v_fullRange_200_);
v_res_204_ = lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__0(v_toApplicative_199_, v_fullRange_boxed_203_, v_val_201_, v_____do__lift_202_);
lean_dec_ref(v_____do__lift_202_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__1(lean_object* v_toApplicative_205_, uint8_t v_fullRange_206_, lean_object* v_toBind_207_, lean_object* v_inst_208_, lean_object* v_____x_209_){
_start:
{
if (lean_obj_tag(v_____x_209_) == 1)
{
lean_object* v_val_210_; lean_object* v___x_211_; lean_object* v___f_212_; lean_object* v___x_213_; 
v_val_210_ = lean_ctor_get(v_____x_209_, 0);
lean_inc(v_val_210_);
lean_dec_ref_known(v_____x_209_, 1);
v___x_211_ = lean_box(v_fullRange_206_);
v___f_212_ = lean_alloc_closure((void*)(lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_212_, 0, v_toApplicative_205_);
lean_closure_set(v___f_212_, 1, v___x_211_);
lean_closure_set(v___f_212_, 2, v_val_210_);
v___x_213_ = lean_apply_4(v_toBind_207_, lean_box(0), lean_box(0), v_inst_208_, v___f_212_);
return v___x_213_;
}
else
{
lean_object* v_toPure_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
lean_dec(v_____x_209_);
lean_dec(v_inst_208_);
lean_dec(v_toBind_207_);
v_toPure_214_ = lean_ctor_get(v_toApplicative_205_, 1);
lean_inc(v_toPure_214_);
lean_dec_ref(v_toApplicative_205_);
v___x_215_ = lean_box(0);
v___x_216_ = lean_apply_2(v_toPure_214_, lean_box(0), v___x_215_);
return v___x_216_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__1___boxed(lean_object* v_toApplicative_217_, lean_object* v_fullRange_218_, lean_object* v_toBind_219_, lean_object* v_inst_220_, lean_object* v_____x_221_){
_start:
{
uint8_t v_fullRange_boxed_222_; lean_object* v_res_223_; 
v_fullRange_boxed_222_ = lean_unbox(v_fullRange_218_);
v_res_223_ = lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__1(v_toApplicative_217_, v_fullRange_boxed_222_, v_toBind_219_, v_inst_220_, v_____x_221_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__2(lean_object* v_decl_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_toBind_228_, lean_object* v___f_229_, lean_object* v_toApplicative_230_, lean_object* v_____do__lift_231_){
_start:
{
uint8_t v___x_232_; 
v___x_232_ = l_Lean_Environment_isImportedConst(v_____do__lift_231_, v_decl_224_);
if (v___x_232_ == 0)
{
lean_object* v___x_233_; lean_object* v___x_234_; 
lean_dec_ref(v_toApplicative_230_);
v___x_233_ = l_Lean_findDeclarationRanges_x3f___redArg(v_inst_225_, v_inst_226_, v_inst_227_, v_decl_224_);
v___x_234_ = lean_apply_4(v_toBind_228_, lean_box(0), lean_box(0), v___x_233_, v___f_229_);
return v___x_234_;
}
else
{
lean_object* v_toPure_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
lean_dec(v___f_229_);
lean_dec(v_toBind_228_);
lean_dec(v_inst_227_);
lean_dec_ref(v_inst_226_);
lean_dec_ref(v_inst_225_);
lean_dec(v_decl_224_);
v_toPure_235_ = lean_ctor_get(v_toApplicative_230_, 1);
lean_inc(v_toPure_235_);
lean_dec_ref(v_toApplicative_230_);
v___x_236_ = lean_box(0);
v___x_237_ = lean_apply_2(v_toPure_235_, lean_box(0), v___x_236_);
return v___x_237_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__2___boxed(lean_object* v_decl_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_toBind_242_, lean_object* v___f_243_, lean_object* v_toApplicative_244_, lean_object* v_____do__lift_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__2(v_decl_238_, v_inst_239_, v_inst_240_, v_inst_241_, v_toBind_242_, v___f_243_, v_toApplicative_244_, v_____do__lift_245_);
lean_dec_ref(v_____do__lift_245_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg(lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_decl_251_, uint8_t v_fullRange_252_){
_start:
{
lean_object* v_toApplicative_253_; lean_object* v_toBind_254_; lean_object* v_getEnv_255_; lean_object* v___x_256_; lean_object* v___f_257_; lean_object* v___f_258_; lean_object* v___x_259_; 
v_toApplicative_253_ = lean_ctor_get(v_inst_247_, 0);
lean_inc_ref_n(v_toApplicative_253_, 2);
v_toBind_254_ = lean_ctor_get(v_inst_247_, 1);
lean_inc_n(v_toBind_254_, 3);
v_getEnv_255_ = lean_ctor_get(v_inst_248_, 0);
lean_inc(v_getEnv_255_);
v___x_256_ = lean_box(v_fullRange_252_);
v___f_257_ = lean_alloc_closure((void*)(lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_257_, 0, v_toApplicative_253_);
lean_closure_set(v___f_257_, 1, v___x_256_);
lean_closure_set(v___f_257_, 2, v_toBind_254_);
lean_closure_set(v___f_257_, 3, v_inst_250_);
v___f_258_ = lean_alloc_closure((void*)(lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___lam__2___boxed), 8, 7);
lean_closure_set(v___f_258_, 0, v_decl_251_);
lean_closure_set(v___f_258_, 1, v_inst_247_);
lean_closure_set(v___f_258_, 2, v_inst_248_);
lean_closure_set(v___f_258_, 3, v_inst_249_);
lean_closure_set(v___f_258_, 4, v_toBind_254_);
lean_closure_set(v___f_258_, 5, v___f_257_);
lean_closure_set(v___f_258_, 6, v_toApplicative_253_);
v___x_259_ = lean_apply_4(v_toBind_254_, lean_box(0), lean_box(0), v_getEnv_255_, v___f_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg___boxed(lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_decl_264_, lean_object* v_fullRange_265_){
_start:
{
uint8_t v_fullRange_boxed_266_; lean_object* v_res_267_; 
v_fullRange_boxed_266_ = lean_unbox(v_fullRange_265_);
v_res_267_ = lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg(v_inst_260_, v_inst_261_, v_inst_262_, v_inst_263_, v_decl_264_, v_fullRange_boxed_266_);
return v_res_267_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f(lean_object* v_m_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_decl_273_, uint8_t v_fullRange_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg(v_inst_269_, v_inst_270_, v_inst_271_, v_inst_272_, v_decl_273_, v_fullRange_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_findDeclarationSyntaxRange_x3f___boxed(lean_object* v_m_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_decl_281_, lean_object* v_fullRange_282_){
_start:
{
uint8_t v_fullRange_boxed_283_; lean_object* v_res_284_; 
v_fullRange_boxed_283_ = lean_unbox(v_fullRange_282_);
v_res_284_ = lp_batteries_Lean_findDeclarationSyntaxRange_x3f(v_m_276_, v_inst_277_, v_inst_278_, v_inst_279_, v_inst_280_, v_decl_281_, v_fullRange_boxed_283_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___lam__0(lean_object* v___x_285_, lean_object* v_withRef_286_, lean_object* v_x_287_, lean_object* v_oldRef_288_){
_start:
{
lean_object* v_ref_289_; lean_object* v___x_290_; 
v_ref_289_ = l_Lean_replaceRef(v___x_285_, v_oldRef_288_);
v___x_290_ = lean_apply_3(v_withRef_286_, lean_box(0), v_ref_289_, v_x_287_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___lam__0___boxed(lean_object* v___x_291_, lean_object* v_withRef_292_, lean_object* v_x_293_, lean_object* v_oldRef_294_){
_start:
{
lean_object* v_res_295_; 
v_res_295_ = lp_batteries_Lean_withDeclRef_x3f___redArg___lam__0(v___x_291_, v_withRef_292_, v_x_293_, v_oldRef_294_);
lean_dec(v_oldRef_294_);
lean_dec(v___x_291_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___lam__1(lean_object* v_inst_296_, uint8_t v_canonical_297_, lean_object* v_x_298_, lean_object* v_toBind_299_, lean_object* v_____x_300_){
_start:
{
if (lean_obj_tag(v_____x_300_) == 1)
{
lean_object* v_val_301_; lean_object* v_getRef_302_; lean_object* v_withRef_303_; lean_object* v___x_304_; lean_object* v___f_305_; lean_object* v___x_306_; 
v_val_301_ = lean_ctor_get(v_____x_300_, 0);
lean_inc(v_val_301_);
lean_dec_ref_known(v_____x_300_, 1);
v_getRef_302_ = lean_ctor_get(v_inst_296_, 0);
lean_inc(v_getRef_302_);
v_withRef_303_ = lean_ctor_get(v_inst_296_, 1);
lean_inc(v_withRef_303_);
lean_dec_ref(v_inst_296_);
v___x_304_ = l_Lean_Syntax_ofRange(v_val_301_, v_canonical_297_);
v___f_305_ = lean_alloc_closure((void*)(lp_batteries_Lean_withDeclRef_x3f___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_305_, 0, v___x_304_);
lean_closure_set(v___f_305_, 1, v_withRef_303_);
lean_closure_set(v___f_305_, 2, v_x_298_);
v___x_306_ = lean_apply_4(v_toBind_299_, lean_box(0), lean_box(0), v_getRef_302_, v___f_305_);
return v___x_306_;
}
else
{
lean_dec(v_____x_300_);
lean_dec(v_toBind_299_);
lean_dec_ref(v_inst_296_);
return v_x_298_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___lam__1___boxed(lean_object* v_inst_307_, lean_object* v_canonical_308_, lean_object* v_x_309_, lean_object* v_toBind_310_, lean_object* v_____x_311_){
_start:
{
uint8_t v_canonical_boxed_312_; lean_object* v_res_313_; 
v_canonical_boxed_312_ = lean_unbox(v_canonical_308_);
v_res_313_ = lp_batteries_Lean_withDeclRef_x3f___redArg___lam__1(v_inst_307_, v_canonical_boxed_312_, v_x_309_, v_toBind_310_, v_____x_311_);
return v_res_313_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg(lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_decl_319_, lean_object* v_x_320_, uint8_t v_fullRange_321_, uint8_t v_canonical_322_){
_start:
{
lean_object* v_toBind_323_; lean_object* v___x_324_; lean_object* v___f_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v_toBind_323_ = lean_ctor_get(v_inst_314_, 1);
lean_inc_n(v_toBind_323_, 2);
v___x_324_ = lean_box(v_canonical_322_);
v___f_325_ = lean_alloc_closure((void*)(lp_batteries_Lean_withDeclRef_x3f___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_325_, 0, v_inst_318_);
lean_closure_set(v___f_325_, 1, v___x_324_);
lean_closure_set(v___f_325_, 2, v_x_320_);
lean_closure_set(v___f_325_, 3, v_toBind_323_);
v___x_326_ = lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg(v_inst_314_, v_inst_315_, v_inst_316_, v_inst_317_, v_decl_319_, v_fullRange_321_);
v___x_327_ = lean_apply_4(v_toBind_323_, lean_box(0), lean_box(0), v___x_326_, v___f_325_);
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___redArg___boxed(lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_decl_333_, lean_object* v_x_334_, lean_object* v_fullRange_335_, lean_object* v_canonical_336_){
_start:
{
uint8_t v_fullRange_boxed_337_; uint8_t v_canonical_boxed_338_; lean_object* v_res_339_; 
v_fullRange_boxed_337_ = lean_unbox(v_fullRange_335_);
v_canonical_boxed_338_ = lean_unbox(v_canonical_336_);
v_res_339_ = lp_batteries_Lean_withDeclRef_x3f___redArg(v_inst_328_, v_inst_329_, v_inst_330_, v_inst_331_, v_inst_332_, v_decl_333_, v_x_334_, v_fullRange_boxed_337_, v_canonical_boxed_338_);
return v_res_339_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f(lean_object* v_00_u03b1_340_, lean_object* v_m_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_decl_347_, lean_object* v_x_348_, uint8_t v_fullRange_349_, uint8_t v_canonical_350_){
_start:
{
lean_object* v_toBind_351_; lean_object* v___x_352_; lean_object* v___f_353_; lean_object* v___x_354_; lean_object* v___x_355_; 
v_toBind_351_ = lean_ctor_get(v_inst_342_, 1);
lean_inc_n(v_toBind_351_, 2);
v___x_352_ = lean_box(v_canonical_350_);
v___f_353_ = lean_alloc_closure((void*)(lp_batteries_Lean_withDeclRef_x3f___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_353_, 0, v_inst_346_);
lean_closure_set(v___f_353_, 1, v___x_352_);
lean_closure_set(v___f_353_, 2, v_x_348_);
lean_closure_set(v___f_353_, 3, v_toBind_351_);
v___x_354_ = lp_batteries_Lean_findDeclarationSyntaxRange_x3f___redArg(v_inst_342_, v_inst_343_, v_inst_344_, v_inst_345_, v_decl_347_, v_fullRange_349_);
v___x_355_ = lean_apply_4(v_toBind_351_, lean_box(0), lean_box(0), v___x_354_, v___f_353_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_withDeclRef_x3f___boxed(lean_object* v_00_u03b1_356_, lean_object* v_m_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_decl_363_, lean_object* v_x_364_, lean_object* v_fullRange_365_, lean_object* v_canonical_366_){
_start:
{
uint8_t v_fullRange_boxed_367_; uint8_t v_canonical_boxed_368_; lean_object* v_res_369_; 
v_fullRange_boxed_367_ = lean_unbox(v_fullRange_365_);
v_canonical_boxed_368_ = lean_unbox(v_canonical_366_);
v_res_369_ = lp_batteries_Lean_withDeclRef_x3f(v_00_u03b1_356_, v_m_357_, v_inst_358_, v_inst_359_, v_inst_360_, v_inst_361_, v_inst_362_, v_decl_363_, v_x_364_, v_fullRange_boxed_367_, v_canonical_boxed_368_);
return v_res_369_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Syntax(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_Lsp_Utf16(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Position(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_Lsp_Utf16(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Position(uint8_t builtin) {
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
lean_object* initialize_Lean_Syntax(uint8_t builtin);
lean_object* initialize_Lean_Data_Lsp_Utf16(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Position(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Syntax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_Lsp_Utf16(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Position(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Position(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Position(builtin);
}
#ifdef __cplusplus
}
#endif
