// Lean compiler output
// Module: Mathlib.Tactic.CrossRefAttribute
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public import Mathlib.Init
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
lean_object* l_instDecidableEqChar___boxed(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Parser_mkAntiquot(lean_object*, lean_object*, uint8_t, uint8_t);
uint8_t lean_uint32_dec_le(uint32_t, uint32_t);
lean_object* l_Lean_Parser_takeWhileFn(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Parser_instBEqError_beq(lean_object*, lean_object*);
lean_object* l_Lean_Parser_ParserState_mkUnexpectedError(lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* l_Lean_Parser_mkNodeToken(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Parser_ParserState_mkError(lean_object*, lean_object*);
lean_object* l_Lean_Parser_mkAtomicInfo(lean_object*);
lean_object* l_Lean_Parser_withAntiquot(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_isLit_x3f(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_findDocString_x3f(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_nextn(lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
extern lean_object* l_Lean_docStringExt;
lean_object* l_String_removeLeadingSpaces(lean_object*);
lean_object* l_Lean_MapDeclarationExtension_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_registerSimplePersistentEnvExtension___redArg(lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_TSyntax_getString(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_data(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t lean_string_hash(lean_object*);
lean_object* l_Lean_Parser_mkAntiquot_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Parenthesizer_withAntiquot_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_instInhabitedConstantInfo_default;
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Parser_mkAntiquot_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Formatter_visitAtom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Formatter_orelse_formatter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic___closed__0_value;
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic_hash(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic_hash___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic_ord(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic_ord___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic_ord___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_PiBaseTopic_urlSubdomain___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "topology"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_PiBaseTopic_urlSubdomain___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_PiBaseTopic_urlSubdomain___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_PiBaseTopic_urlSubdomain(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_PiBaseTopic_label___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Topology"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_PiBaseTopic_label___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_PiBaseTopic_label___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_PiBaseTopic_label(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_PiBaseTopic_shortName(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_dlmf_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_dlmf_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_kerodon_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_kerodon_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_lmfdb_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_lmfdb_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_pibase_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_pibase_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_stacks_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_stacks_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_wikidata_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_wikidata_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instBEqDatabase_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instBEqDatabase_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_instBEqDatabase___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_instBEqDatabase_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_instBEqDatabase___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instBEqDatabase___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_instBEqDatabase = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instBEqDatabase___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___closed__0;
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_instHashableDatabase___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_instHashableDatabase___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instHashableDatabase___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_instHashableDatabase = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instHashableDatabase___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instOrdDatabase_ord(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instOrdDatabase_ord___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_instOrdDatabase___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_instOrdDatabase_ord___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_instOrdDatabase___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instOrdDatabase___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_instOrdDatabase = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instOrdDatabase___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "https://topology.pi-base.org/"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "/"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "https://dlmf.nist.gov/"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "https://kerodon.net/tag/"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "https://www.lmfdb.org/knowledge/show/"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "P"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "S"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "T"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "theorems"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "spaces"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "properties"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "https://stacks.math.columbia.edu/tag/"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_url___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "https://www.wikidata.org/wiki/"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_url(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_label___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "DLMF"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_label___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_label___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_label___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Kerodon Tag"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_label___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_label___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_label___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "LMFDB"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_label___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_label___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_label___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 17, .m_data = "π-Base (Topology)"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_label___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_label___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_label___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Stacks Tag"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_label___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_label___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_label___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Wikidata"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_label___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_label___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_label(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_label___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "dlmf"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "kerodon"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lmfdb"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "pibase-topology"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "stacks"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "wikidata"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instBEqTag_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instBEqTag_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_instBEqTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_instBEqTag_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_instBEqTag___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instBEqTag___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_instBEqTag = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instBEqTag___closed__0_value;
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_CrossRef_instHashableTag_hash(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instHashableTag_hash___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_instHashableTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_instHashableTag_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_instHashableTag___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instHashableTag___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_instHashableTag = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_instHashableTag___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "CrossRef"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tagExt"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(77, 155, 146, 187, 35, 30, 245, 72)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_tagExt;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_CrossRef_addCrossRefDoc_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "invalid doc string, declaration `"};
static const lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__1;
static const lean_string_object lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "` is in an imported module"};
static const lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "]("};
static const lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "stacksTag"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0_value),LEAN_SCALAR_PTR_LITERAL(179, 249, 177, 185, 212, 207, 173, 126)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagKind = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_stacksTagFn___lam__0(uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagFn___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_CrossRef_stacksTagFn_spec__1(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_CrossRef_stacksTagFn_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_stacksTagFn___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "Stacks tags must consist only of digits and uppercase letters."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Stacks tags must be exactly 4 characters"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "stacks tag"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagFn(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "wikidataId"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__0_value),LEAN_SCALAR_PTR_LITERAL(122, 158, 89, 214, 161, 46, 75, 228)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdKind = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_List_all___at___00Mathlib_CrossRef_wikidataIdFn_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_wikidataIdFn_spec__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 74, .m_capacity = 74, .m_length = 73, .m_data = "Wikidata ids must start with the letter Q followed by one or more digits."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "Wikidata ids must consist of the letter Q followed by digits."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "wikidata id"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdFn(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "lmfdbId"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 213, 18, 114, 24, 187, 81, 53)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdKind = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___lam__0(uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_all___at___00Mathlib_CrossRef_lmfdbIdFn_spec__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_lmfdbIdFn_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 84, .m_capacity = 84, .m_length = 83, .m_data = "LMFDB ids must consist only of lowercase letters, digits, periods, and underscores."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "lmfdb id"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdFn(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "pibaseId"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__0_value),LEAN_SCALAR_PTR_LITERAL(127, 99, 153, 66, 54, 9, 244, 5)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdKind = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 38, .m_data = "π-Base ids must start with P, S, or T."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 52, .m_data = "π-Base ids must have exactly six digits after P/S/T."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 61, .m_data = "π-Base ids must consist of P, S, or T followed by six digits."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 9, .m_data = "π-Base id"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdFn(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "dlmfId"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__0_value),LEAN_SCALAR_PTR_LITERAL(221, 230, 82, 173, 186, 221, 150, 134)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdKind = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1_value;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12___boxed__const__1;
static lean_once_cell_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12;
LEAN_EXPORT uint8_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_dlmfIdFn___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 118, .m_capacity = 118, .m_length = 117, .m_data = "DLMF references must consist only of (lowercase) roman numerals, the letters E/T/F, digits, periods, and underscores."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdFn___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdFn___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_dlmfIdFn___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "dlmf id"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdFn___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdFn___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdFn(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser;
static const lean_string_object lp_mathlib_Lean_TSyntax_getStacksTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Malformed Stacks tag"};
static const lean_object* lp_mathlib_Lean_TSyntax_getStacksTag___closed__0 = (const lean_object*)&lp_mathlib_Lean_TSyntax_getStacksTag___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_TSyntax_getStacksTag___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_TSyntax_getStacksTag___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getStacksTag(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getStacksTag___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_TSyntax_getWikidataId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Malformed Wikidata id"};
static const lean_object* lp_mathlib_Lean_TSyntax_getWikidataId___closed__0 = (const lean_object*)&lp_mathlib_Lean_TSyntax_getWikidataId___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_TSyntax_getWikidataId___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_TSyntax_getWikidataId___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getWikidataId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getWikidataId___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_TSyntax_getLmfdbId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Malformed LMFDB id"};
static const lean_object* lp_mathlib_Lean_TSyntax_getLmfdbId___closed__0 = (const lean_object*)&lp_mathlib_Lean_TSyntax_getLmfdbId___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_TSyntax_getLmfdbId___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_TSyntax_getLmfdbId___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getLmfdbId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getLmfdbId___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_TSyntax_getPibaseId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 19, .m_data = "Malformed π-Base id"};
static const lean_object* lp_mathlib_Lean_TSyntax_getPibaseId___closed__0 = (const lean_object*)&lp_mathlib_Lean_TSyntax_getPibaseId___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_TSyntax_getPibaseId___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_TSyntax_getPibaseId___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getPibaseId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getPibaseId___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_TSyntax_getDlmfId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Malformed DLMF ref."};
static const lean_object* lp_mathlib_Lean_TSyntax_getDlmfId___closed__0 = (const lean_object*)&lp_mathlib_Lean_TSyntax_getDlmfId___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_TSyntax_getDlmfId___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_TSyntax_getDlmfId___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getDlmfId(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getDlmfId___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_stacksTagNoAntiquot_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_stacksTagNoAntiquot_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_wikidataIdNoAntiquot_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_wikidataIdNoAntiquot_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_lmfdbIdNoAntiquot_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_lmfdbIdNoAntiquot_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_pibaseIdNoAntiquot_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_pibaseIdNoAntiquot_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_dlmfIdNoAntiquot_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_dlmfIdNoAntiquot_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "quot"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(145, 163, 173, 41, 168, 168, 65, 81)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "stacksTagDB"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(203, 237, 210, 16, 234, 16, 200, 169)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__3_value),LEAN_SCALAR_PTR_LITERAL(17, 195, 246, 230, 199, 202, 242, 43)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__7_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "`(stacksTagDB| "};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__5_value),LEAN_SCALAR_PTR_LITERAL(203, 237, 210, 16, 234, 16, 200, 169)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__6_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__4_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Parser_Category_stacksTagDB;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "stacksTagDBKerodon"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 236, 73, 87, 185, 150, 114, 66)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "stacksTagDBStacks"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 47, 20, 107, 27, 7, 217, 92)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__0_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 103, 9, 114, 203, 121, 69, 78)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "stacksTagParser"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__2_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__1_value),LEAN_SCALAR_PTR_LITERAL(25, 200, 251, 144, 124, 37, 4, 218)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__7_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__10_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTag___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__0_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTag = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__16_value;
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_formatter___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_parenthesizer___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "CrossRefAttribute"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(180, 169, 36, 148, 137, 249, 0, 199)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(53, 92, 13, 87, 217, 123, 238, 235)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(208, 247, 134, 240, 105, 152, 213, 86)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(237, 22, 212, 115, 45, 143, 29, 75)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__10_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__10_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__10_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__11_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__10_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(196, 0, 244, 245, 77, 32, 43, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__11_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__11_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__12_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__12_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__12_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__13_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__11_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__12_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(77, 124, 58, 2, 188, 179, 252, 208)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__13_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__13_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__14_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__13_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(8, 227, 228, 157, 50, 143, 89, 191)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__14_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__14_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__15_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__14_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(245, 232, 73, 91, 127, 189, 61, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__15_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__15_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__16_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__15_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 178, 172, 40, 45, 109, 59, 9)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__16_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__16_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__17_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__16_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2143948442) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(201, 95, 166, 65, 48, 35, 231, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__17_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__17_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__18_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__18_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__18_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__19_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__17_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__18_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(90, 137, 65, 113, 122, 124, 84, 188)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__19_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__19_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__20_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__20_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__20_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__21_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__19_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__20_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(38, 110, 39, 89, 88, 76, 244, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__21_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__21_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__22_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__21_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(103, 111, 20, 229, 70, 87, 237, 78)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__22_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__22_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__23_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed, .m_arity = 11, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__23_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__23_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__24_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__24_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__24_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__25_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Apply a Stacks or Kerodon project tag to a theorem."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__25_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__25_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__26_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__22_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__25_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(2, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__26_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__26_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__27_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__26_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__23_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__24_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__27_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__27_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "wikidataTag"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__0_value),LEAN_SCALAR_PTR_LITERAL(110, 11, 241, 34, 249, 78, 97, 26)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "wikidataIdParser"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__3_value),LEAN_SCALAR_PTR_LITERAL(184, 236, 57, 32, 188, 180, 27, 209)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTag = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__8_value;
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_formatter___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_parenthesizer___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2____boxed, .m_arity = 11, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__0_value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTag___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 194, 32, 14, 100, 80, 12, 56)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "Apply a Wikidata identifier to a declaration."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "lmfdbTag"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__0_value),LEAN_SCALAR_PTR_LITERAL(239, 153, 66, 27, 109, 11, 26, 214)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "lmfdbIdParser"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__3_value),LEAN_SCALAR_PTR_LITERAL(132, 3, 216, 68, 92, 63, 202, 245)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTag = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__8_value;
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_formatter___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_parenthesizer___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2____boxed, .m_arity = 11, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__0_value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTag___closed__0_value),LEAN_SCALAR_PTR_LITERAL(63, 186, 206, 223, 146, 194, 145, 203)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "Apply an LMFDB identifier to a declaration."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "pibaseTopic"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 140, 247, 101, 184, 67, 78, 230)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_PiBaseTopic_urlSubdomain___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__3_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTopic = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_getPiBaseTopic_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_getPiBaseTopic_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_getPiBaseTopic_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_getPiBaseTopic_x3f(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "pibaseTag"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__0_value),LEAN_SCALAR_PTR_LITERAL(53, 15, 147, 142, 177, 37, 193, 116)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "pibase"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "pibaseIdParser"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__5_value),LEAN_SCALAR_PTR_LITERAL(111, 95, 147, 159, 68, 254, 20, 100)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTag = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__10_value;
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_formatter___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_parenthesizer___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2____boxed, .m_arity = 11, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__0_value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTag___closed__0_value),LEAN_SCALAR_PTR_LITERAL(165, 158, 75, 115, 18, 205, 105, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 43, .m_data = "Apply a π-Base identifier to a declaration."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2____boxed(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "dlmfTag"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__0_value),LEAN_SCALAR_PTR_LITERAL(79, 16, 176, 34, 60, 167, 145, 130)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "dlmfIdParser"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__3_value),LEAN_SCALAR_PTR_LITERAL(193, 44, 147, 84, 31, 163, 130, 185)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTag = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__8_value;
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_formatter___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Parser_mkAntiquot_parenthesizer___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__16_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1047927000) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(111, 235, 241, 71, 106, 83, 9, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__18_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(100, 216, 16, 235, 18, 195, 214, 28)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__20_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(240, 152, 149, 113, 199, 61, 170, 249)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(161, 189, 247, 96, 103, 204, 112, 175)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2____boxed, .m_arity = 11, .m_num_fixed = 5, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__0_value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTag___closed__0_value),LEAN_SCALAR_PTR_LITERAL(223, 91, 175, 101, 170, 124, 185, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "Apply a DLMF identifier to a declaration."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(2, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__0;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__1;
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_CrossRef_traceCrossRefs_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_CrossRef_traceCrossRefs_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_Database_url___closed__8_value)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__2;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__3;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__4;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = ") corresponds to declaration '"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__6;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "'."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__8;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__1;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__4;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "No tags found."};
static const lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__7;
static const lean_array_object lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "stacksTags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__0_value),LEAN_SCALAR_PTR_LITERAL(64, 83, 93, 136, 86, 21, 233, 209)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "#stacks_tags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_stacksTags___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_stacksTags = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "kerodonTags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 171, 197, 121, 36, 146, 222, 171)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "#kerodon_tags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_kerodonTags = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__kerodonTags__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__kerodonTags__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "wikidataTags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__0_value),LEAN_SCALAR_PTR_LITERAL(215, 218, 145, 107, 128, 72, 226, 148)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "#wikidata_tags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_wikidataTags = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__wikidataTags__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__wikidataTags__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "lmfdbTags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__0_value),LEAN_SCALAR_PTR_LITERAL(42, 38, 212, 109, 86, 241, 207, 56)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "#lmfdb_tags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbTags = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__lmfdbTags__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__lmfdbTags__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "pibaseTags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__0_value),LEAN_SCALAR_PTR_LITERAL(190, 250, 200, 25, 168, 202, 37, 140)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "#pibase_tags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__5_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTag___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_pibaseTags = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__pibaseTags__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__pibaseTags__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "dlmfTags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(75, 93, 28, 21, 13, 247, 66, 178)}};
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__0_value),LEAN_SCALAR_PTR_LITERAL(123, 204, 170, 128, 70, 135, 181, 173)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "#dlmf_tags"};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTagDB_quot___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_CrossRef_stacksTags___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_CrossRef_dlmfTags = (const lean_object*)&lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__dlmfTags__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__dlmfTags__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic_beq(lean_object* v_x_1_, lean_object* v_y_2_){
_start:
{
uint8_t v___x_3_; 
v___x_3_ = 1;
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic_beq___boxed(lean_object* v_x_4_, lean_object* v_y_5_){
_start:
{
uint8_t v_res_6_; lean_object* v_r_7_; 
v_res_6_ = lp_mathlib_Mathlib_CrossRef_instBEqPiBaseTopic_beq(v_x_4_, v_y_5_);
v_r_7_ = lean_box(v_res_6_);
return v_r_7_;
}
}
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic_hash(lean_object* v_x_10_){
_start:
{
uint64_t v___x_11_; 
v___x_11_ = 0ULL;
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic_hash___boxed(lean_object* v_x_12_){
_start:
{
uint64_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_mathlib_Mathlib_CrossRef_instHashablePiBaseTopic_hash(v_x_12_);
v_r_14_ = lean_box_uint64(v_res_13_);
return v_r_14_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic_ord(lean_object* v_x_17_, lean_object* v_y_18_){
_start:
{
uint8_t v___x_19_; 
v___x_19_ = 1;
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic_ord___boxed(lean_object* v_x_20_, lean_object* v_y_21_){
_start:
{
uint8_t v_res_22_; lean_object* v_r_23_; 
v_res_22_ = lp_mathlib_Mathlib_CrossRef_instOrdPiBaseTopic_ord(v_x_20_, v_y_21_);
v_r_23_ = lean_box(v_res_22_);
return v_r_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_PiBaseTopic_urlSubdomain(lean_object* v_x_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_PiBaseTopic_urlSubdomain___closed__0));
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_PiBaseTopic_label(lean_object* v_x_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_PiBaseTopic_label___closed__0));
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_PiBaseTopic_shortName(lean_object* v_x_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_PiBaseTopic_urlSubdomain___closed__0));
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorIdx(lean_object* v_x_34_){
_start:
{
switch(lean_obj_tag(v_x_34_))
{
case 0:
{
lean_object* v___x_35_; 
v___x_35_ = lean_unsigned_to_nat(0u);
return v___x_35_;
}
case 1:
{
lean_object* v___x_36_; 
v___x_36_ = lean_unsigned_to_nat(1u);
return v___x_36_;
}
case 2:
{
lean_object* v___x_37_; 
v___x_37_ = lean_unsigned_to_nat(2u);
return v___x_37_;
}
case 3:
{
lean_object* v___x_38_; 
v___x_38_ = lean_unsigned_to_nat(3u);
return v___x_38_;
}
case 4:
{
lean_object* v___x_39_; 
v___x_39_ = lean_unsigned_to_nat(4u);
return v___x_39_;
}
default: 
{
lean_object* v___x_40_; 
v___x_40_ = lean_unsigned_to_nat(5u);
return v___x_40_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorIdx___boxed(lean_object* v_x_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_Mathlib_CrossRef_Database_ctorIdx(v_x_41_);
lean_dec(v_x_41_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(lean_object* v_t_43_, lean_object* v_k_44_){
_start:
{
if (lean_obj_tag(v_t_43_) == 3)
{
lean_object* v_topic_45_; lean_object* v___x_46_; 
v_topic_45_ = lean_ctor_get(v_t_43_, 0);
lean_inc(v_topic_45_);
lean_dec_ref_known(v_t_43_, 1);
v___x_46_ = lean_apply_1(v_k_44_, v_topic_45_);
return v___x_46_;
}
else
{
lean_dec(v_t_43_);
return v_k_44_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorElim(lean_object* v_motive_47_, lean_object* v_ctorIdx_48_, lean_object* v_t_49_, lean_object* v_h_50_, lean_object* v_k_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_49_, v_k_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_ctorElim___boxed(lean_object* v_motive_53_, lean_object* v_ctorIdx_54_, lean_object* v_t_55_, lean_object* v_h_56_, lean_object* v_k_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim(v_motive_53_, v_ctorIdx_54_, v_t_55_, v_h_56_, v_k_57_);
lean_dec(v_ctorIdx_54_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_dlmf_elim___redArg(lean_object* v_t_59_, lean_object* v_dlmf_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_59_, v_dlmf_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_dlmf_elim(lean_object* v_motive_62_, lean_object* v_t_63_, lean_object* v_h_64_, lean_object* v_dlmf_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_63_, v_dlmf_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_kerodon_elim___redArg(lean_object* v_t_67_, lean_object* v_kerodon_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_67_, v_kerodon_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_kerodon_elim(lean_object* v_motive_70_, lean_object* v_t_71_, lean_object* v_h_72_, lean_object* v_kerodon_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_71_, v_kerodon_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_lmfdb_elim___redArg(lean_object* v_t_75_, lean_object* v_lmfdb_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_75_, v_lmfdb_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_lmfdb_elim(lean_object* v_motive_78_, lean_object* v_t_79_, lean_object* v_h_80_, lean_object* v_lmfdb_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_79_, v_lmfdb_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_pibase_elim___redArg(lean_object* v_t_83_, lean_object* v_pibase_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_83_, v_pibase_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_pibase_elim(lean_object* v_motive_86_, lean_object* v_t_87_, lean_object* v_h_88_, lean_object* v_pibase_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_87_, v_pibase_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_stacks_elim___redArg(lean_object* v_t_91_, lean_object* v_stacks_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_91_, v_stacks_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_stacks_elim(lean_object* v_motive_94_, lean_object* v_t_95_, lean_object* v_h_96_, lean_object* v_stacks_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_95_, v_stacks_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_wikidata_elim___redArg(lean_object* v_t_99_, lean_object* v_wikidata_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_99_, v_wikidata_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_wikidata_elim(lean_object* v_motive_102_, lean_object* v_t_103_, lean_object* v_h_104_, lean_object* v_wikidata_105_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_mathlib_Mathlib_CrossRef_Database_ctorElim___redArg(v_t_103_, v_wikidata_105_);
return v___x_106_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instBEqDatabase_beq(lean_object* v_x_107_, lean_object* v_x_108_){
_start:
{
switch(lean_obj_tag(v_x_107_))
{
case 0:
{
if (lean_obj_tag(v_x_108_) == 0)
{
uint8_t v___x_109_; 
v___x_109_ = 1;
return v___x_109_;
}
else
{
uint8_t v___x_110_; 
v___x_110_ = 0;
return v___x_110_;
}
}
case 1:
{
if (lean_obj_tag(v_x_108_) == 1)
{
uint8_t v___x_111_; 
v___x_111_ = 1;
return v___x_111_;
}
else
{
uint8_t v___x_112_; 
v___x_112_ = 0;
return v___x_112_;
}
}
case 2:
{
if (lean_obj_tag(v_x_108_) == 2)
{
uint8_t v___x_113_; 
v___x_113_ = 1;
return v___x_113_;
}
else
{
uint8_t v___x_114_; 
v___x_114_ = 0;
return v___x_114_;
}
}
case 3:
{
if (lean_obj_tag(v_x_108_) == 3)
{
uint8_t v___x_115_; 
v___x_115_ = 1;
return v___x_115_;
}
else
{
uint8_t v___x_116_; 
v___x_116_ = 0;
return v___x_116_;
}
}
case 4:
{
if (lean_obj_tag(v_x_108_) == 4)
{
uint8_t v___x_117_; 
v___x_117_ = 1;
return v___x_117_;
}
else
{
uint8_t v___x_118_; 
v___x_118_ = 0;
return v___x_118_;
}
}
default: 
{
if (lean_obj_tag(v_x_108_) == 5)
{
uint8_t v___x_119_; 
v___x_119_ = 1;
return v___x_119_;
}
else
{
uint8_t v___x_120_; 
v___x_120_ = 0;
return v___x_120_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instBEqDatabase_beq___boxed(lean_object* v_x_121_, lean_object* v_x_122_){
_start:
{
uint8_t v_res_123_; lean_object* v_r_124_; 
v_res_123_ = lp_mathlib_Mathlib_CrossRef_instBEqDatabase_beq(v_x_121_, v_x_122_);
lean_dec(v_x_122_);
lean_dec(v_x_121_);
v_r_124_ = lean_box(v_res_123_);
return v_r_124_;
}
}
static uint64_t _init_lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___closed__0(void){
_start:
{
uint64_t v___x_127_; uint64_t v___x_128_; uint64_t v___x_129_; 
v___x_127_ = 0ULL;
v___x_128_ = 3ULL;
v___x_129_ = lean_uint64_mix_hash(v___x_128_, v___x_127_);
return v___x_129_;
}
}
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash(lean_object* v_x_130_){
_start:
{
switch(lean_obj_tag(v_x_130_))
{
case 0:
{
uint64_t v___x_131_; 
v___x_131_ = 0ULL;
return v___x_131_;
}
case 1:
{
uint64_t v___x_132_; 
v___x_132_ = 1ULL;
return v___x_132_;
}
case 2:
{
uint64_t v___x_133_; 
v___x_133_ = 2ULL;
return v___x_133_;
}
case 3:
{
uint64_t v___x_134_; 
v___x_134_ = lean_uint64_once(&lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___closed__0, &lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___closed__0);
return v___x_134_;
}
case 4:
{
uint64_t v___x_135_; 
v___x_135_ = 4ULL;
return v___x_135_;
}
default: 
{
uint64_t v___x_136_; 
v___x_136_ = 5ULL;
return v___x_136_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash___boxed(lean_object* v_x_137_){
_start:
{
uint64_t v_res_138_; lean_object* v_r_139_; 
v_res_138_ = lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash(v_x_137_);
lean_dec(v_x_137_);
v_r_139_ = lean_box_uint64(v_res_138_);
return v_r_139_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instOrdDatabase_ord(lean_object* v_x_142_, lean_object* v_x_143_){
_start:
{
switch(lean_obj_tag(v_x_142_))
{
case 0:
{
switch(lean_obj_tag(v_x_143_))
{
case 0:
{
uint8_t v___x_144_; 
v___x_144_ = 1;
return v___x_144_;
}
case 1:
{
uint8_t v___x_145_; 
v___x_145_ = 0;
return v___x_145_;
}
case 2:
{
uint8_t v___x_146_; 
v___x_146_ = 0;
return v___x_146_;
}
case 3:
{
uint8_t v___x_147_; 
v___x_147_ = 0;
return v___x_147_;
}
case 4:
{
uint8_t v___x_148_; 
v___x_148_ = 0;
return v___x_148_;
}
default: 
{
uint8_t v___x_149_; 
v___x_149_ = 0;
return v___x_149_;
}
}
}
case 1:
{
switch(lean_obj_tag(v_x_143_))
{
case 0:
{
uint8_t v___x_150_; 
v___x_150_ = 2;
return v___x_150_;
}
case 1:
{
uint8_t v___x_151_; 
v___x_151_ = 1;
return v___x_151_;
}
case 2:
{
uint8_t v___x_152_; 
v___x_152_ = 0;
return v___x_152_;
}
case 3:
{
uint8_t v___x_153_; 
v___x_153_ = 0;
return v___x_153_;
}
case 4:
{
uint8_t v___x_154_; 
v___x_154_ = 0;
return v___x_154_;
}
default: 
{
uint8_t v___x_155_; 
v___x_155_ = 0;
return v___x_155_;
}
}
}
case 2:
{
switch(lean_obj_tag(v_x_143_))
{
case 0:
{
uint8_t v___x_156_; 
v___x_156_ = 2;
return v___x_156_;
}
case 1:
{
uint8_t v___x_157_; 
v___x_157_ = 2;
return v___x_157_;
}
case 2:
{
uint8_t v___x_158_; 
v___x_158_ = 1;
return v___x_158_;
}
case 3:
{
uint8_t v___x_159_; 
v___x_159_ = 0;
return v___x_159_;
}
case 4:
{
uint8_t v___x_160_; 
v___x_160_ = 0;
return v___x_160_;
}
default: 
{
uint8_t v___x_161_; 
v___x_161_ = 0;
return v___x_161_;
}
}
}
case 3:
{
switch(lean_obj_tag(v_x_143_))
{
case 0:
{
uint8_t v___x_162_; 
v___x_162_ = 2;
return v___x_162_;
}
case 1:
{
uint8_t v___x_163_; 
v___x_163_ = 2;
return v___x_163_;
}
case 2:
{
uint8_t v___x_164_; 
v___x_164_ = 2;
return v___x_164_;
}
case 3:
{
uint8_t v___x_165_; 
v___x_165_ = 1;
return v___x_165_;
}
case 4:
{
uint8_t v___x_166_; 
v___x_166_ = 0;
return v___x_166_;
}
default: 
{
uint8_t v___x_167_; 
v___x_167_ = 0;
return v___x_167_;
}
}
}
case 4:
{
switch(lean_obj_tag(v_x_143_))
{
case 0:
{
uint8_t v___x_168_; 
v___x_168_ = 2;
return v___x_168_;
}
case 1:
{
uint8_t v___x_169_; 
v___x_169_ = 2;
return v___x_169_;
}
case 2:
{
uint8_t v___x_170_; 
v___x_170_ = 2;
return v___x_170_;
}
case 3:
{
uint8_t v___x_171_; 
v___x_171_ = 2;
return v___x_171_;
}
case 4:
{
uint8_t v___x_172_; 
v___x_172_ = 1;
return v___x_172_;
}
default: 
{
uint8_t v___x_173_; 
v___x_173_ = 0;
return v___x_173_;
}
}
}
default: 
{
if (lean_obj_tag(v_x_143_) == 5)
{
uint8_t v___x_174_; 
v___x_174_ = 1;
return v___x_174_;
}
else
{
uint8_t v___x_175_; 
v___x_175_ = 2;
return v___x_175_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instOrdDatabase_ord___boxed(lean_object* v_x_176_, lean_object* v_x_177_){
_start:
{
uint8_t v_res_178_; lean_object* v_r_179_; 
v_res_178_ = lp_mathlib_Mathlib_CrossRef_instOrdDatabase_ord(v_x_176_, v_x_177_);
lean_dec(v_x_177_);
lean_dec(v_x_176_);
v_r_179_ = lean_box(v_res_178_);
return v_r_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_url(lean_object* v_x_196_, lean_object* v_x_197_){
_start:
{
lean_object* v___y_199_; 
switch(lean_obj_tag(v_x_196_))
{
case 0:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_205_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__2));
v___x_206_ = lean_string_append(v___x_205_, v_x_197_);
lean_dec_ref(v_x_197_);
return v___x_206_;
}
case 1:
{
lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_207_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__3));
v___x_208_ = lean_string_append(v___x_207_, v_x_197_);
lean_dec_ref(v_x_197_);
return v___x_208_;
}
case 2:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__4));
v___x_210_ = lean_string_append(v___x_209_, v_x_197_);
lean_dec_ref(v_x_197_);
return v___x_210_;
}
case 3:
{
lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; uint8_t v___x_219_; 
v___x_211_ = lean_unsigned_to_nat(1u);
v___x_212_ = lean_unsigned_to_nat(0u);
v___x_213_ = lean_string_utf8_byte_size(v_x_197_);
lean_inc_ref_n(v_x_197_, 2);
v___x_214_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_214_, 0, v_x_197_);
lean_ctor_set(v___x_214_, 1, v___x_212_);
lean_ctor_set(v___x_214_, 2, v___x_213_);
v___x_215_ = l_String_Slice_Pos_nextn(v___x_214_, v___x_212_, v___x_211_);
lean_dec_ref_known(v___x_214_, 3);
v___x_216_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_216_, 0, v_x_197_);
lean_ctor_set(v___x_216_, 1, v___x_212_);
lean_ctor_set(v___x_216_, 2, v___x_215_);
v___x_217_ = l_String_Slice_toString(v___x_216_);
lean_dec_ref_known(v___x_216_, 3);
v___x_218_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__5));
v___x_219_ = lean_string_dec_eq(v___x_217_, v___x_218_);
if (v___x_219_ == 0)
{
lean_object* v___x_220_; uint8_t v___x_221_; 
v___x_220_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__6));
v___x_221_ = lean_string_dec_eq(v___x_217_, v___x_220_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; uint8_t v___x_223_; 
v___x_222_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__7));
v___x_223_ = lean_string_dec_eq(v___x_217_, v___x_222_);
lean_dec_ref(v___x_217_);
if (v___x_223_ == 0)
{
lean_object* v___x_224_; 
v___x_224_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___y_199_ = v___x_224_;
goto v___jp_198_;
}
else
{
lean_object* v___x_225_; 
v___x_225_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__9));
v___y_199_ = v___x_225_;
goto v___jp_198_;
}
}
else
{
lean_object* v___x_226_; 
lean_dec_ref(v___x_217_);
v___x_226_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__10));
v___y_199_ = v___x_226_;
goto v___jp_198_;
}
}
else
{
lean_object* v___x_227_; 
lean_dec_ref(v___x_217_);
v___x_227_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__11));
v___y_199_ = v___x_227_;
goto v___jp_198_;
}
}
case 4:
{
lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_228_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__12));
v___x_229_ = lean_string_append(v___x_228_, v_x_197_);
lean_dec_ref(v_x_197_);
return v___x_229_;
}
default: 
{
lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_230_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__13));
v___x_231_ = lean_string_append(v___x_230_, v_x_197_);
lean_dec_ref(v_x_197_);
return v___x_231_;
}
}
v___jp_198_:
{
lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_200_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__0));
v___x_201_ = lean_string_append(v___x_200_, v___y_199_);
v___x_202_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__1));
v___x_203_ = lean_string_append(v___x_201_, v___x_202_);
v___x_204_ = lean_string_append(v___x_203_, v_x_197_);
lean_dec_ref(v_x_197_);
return v___x_204_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_url___boxed(lean_object* v_x_232_, lean_object* v_x_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_Mathlib_CrossRef_Database_url(v_x_232_, v_x_233_);
lean_dec(v_x_232_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_label(lean_object* v_x_241_){
_start:
{
switch(lean_obj_tag(v_x_241_))
{
case 0:
{
lean_object* v___x_242_; 
v___x_242_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_label___closed__0));
return v___x_242_;
}
case 1:
{
lean_object* v___x_243_; 
v___x_243_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_label___closed__1));
return v___x_243_;
}
case 2:
{
lean_object* v___x_244_; 
v___x_244_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_label___closed__2));
return v___x_244_;
}
case 3:
{
lean_object* v___x_245_; 
v___x_245_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_label___closed__3));
return v___x_245_;
}
case 4:
{
lean_object* v___x_246_; 
v___x_246_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_label___closed__4));
return v___x_246_;
}
default: 
{
lean_object* v___x_247_; 
v___x_247_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_label___closed__5));
return v___x_247_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_label___boxed(lean_object* v_x_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_mathlib_Mathlib_CrossRef_Database_label(v_x_248_);
lean_dec(v_x_248_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName(lean_object* v_x_256_){
_start:
{
switch(lean_obj_tag(v_x_256_))
{
case 0:
{
lean_object* v___x_257_; 
v___x_257_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__0));
return v___x_257_;
}
case 1:
{
lean_object* v___x_258_; 
v___x_258_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__1));
return v___x_258_;
}
case 2:
{
lean_object* v___x_259_; 
v___x_259_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__2));
return v___x_259_;
}
case 3:
{
lean_object* v___x_260_; 
v___x_260_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__3));
return v___x_260_;
}
case 4:
{
lean_object* v___x_261_; 
v___x_261_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__4));
return v___x_261_;
}
default: 
{
lean_object* v___x_262_; 
v___x_262_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_shortName___closed__5));
return v___x_262_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_Database_shortName___boxed(lean_object* v_x_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_Mathlib_CrossRef_Database_shortName(v_x_263_);
lean_dec(v_x_263_);
return v_res_264_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_instBEqTag_beq(lean_object* v_x_265_, lean_object* v_x_266_){
_start:
{
lean_object* v_declName_267_; lean_object* v_database_268_; lean_object* v_tag_269_; lean_object* v_comment_270_; lean_object* v_declName_271_; lean_object* v_database_272_; lean_object* v_tag_273_; lean_object* v_comment_274_; uint8_t v___x_275_; 
v_declName_267_ = lean_ctor_get(v_x_265_, 0);
v_database_268_ = lean_ctor_get(v_x_265_, 1);
v_tag_269_ = lean_ctor_get(v_x_265_, 2);
v_comment_270_ = lean_ctor_get(v_x_265_, 3);
v_declName_271_ = lean_ctor_get(v_x_266_, 0);
v_database_272_ = lean_ctor_get(v_x_266_, 1);
v_tag_273_ = lean_ctor_get(v_x_266_, 2);
v_comment_274_ = lean_ctor_get(v_x_266_, 3);
v___x_275_ = lean_name_eq(v_declName_267_, v_declName_271_);
if (v___x_275_ == 0)
{
return v___x_275_;
}
else
{
uint8_t v___x_276_; 
v___x_276_ = lp_mathlib_Mathlib_CrossRef_instBEqDatabase_beq(v_database_268_, v_database_272_);
if (v___x_276_ == 0)
{
return v___x_276_;
}
else
{
uint8_t v___x_277_; 
v___x_277_ = lean_string_dec_eq(v_tag_269_, v_tag_273_);
if (v___x_277_ == 0)
{
return v___x_277_;
}
else
{
uint8_t v___x_278_; 
v___x_278_ = lean_string_dec_eq(v_comment_270_, v_comment_274_);
return v___x_278_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instBEqTag_beq___boxed(lean_object* v_x_279_, lean_object* v_x_280_){
_start:
{
uint8_t v_res_281_; lean_object* v_r_282_; 
v_res_281_ = lp_mathlib_Mathlib_CrossRef_instBEqTag_beq(v_x_279_, v_x_280_);
lean_dec_ref(v_x_280_);
lean_dec_ref(v_x_279_);
v_r_282_ = lean_box(v_res_281_);
return v_r_282_;
}
}
LEAN_EXPORT uint64_t lp_mathlib_Mathlib_CrossRef_instHashableTag_hash(lean_object* v_x_285_){
_start:
{
lean_object* v_declName_286_; lean_object* v_database_287_; lean_object* v_tag_288_; lean_object* v_comment_289_; uint64_t v___x_290_; uint64_t v___y_292_; 
v_declName_286_ = lean_ctor_get(v_x_285_, 0);
v_database_287_ = lean_ctor_get(v_x_285_, 1);
v_tag_288_ = lean_ctor_get(v_x_285_, 2);
v_comment_289_ = lean_ctor_get(v_x_285_, 3);
v___x_290_ = 0ULL;
if (lean_obj_tag(v_declName_286_) == 0)
{
uint64_t v___x_300_; 
v___x_300_ = 1723ULL;
v___y_292_ = v___x_300_;
goto v___jp_291_;
}
else
{
uint64_t v_hash_301_; 
v_hash_301_ = lean_ctor_get_uint64(v_declName_286_, sizeof(void*)*2);
v___y_292_ = v_hash_301_;
goto v___jp_291_;
}
v___jp_291_:
{
uint64_t v___x_293_; uint64_t v___x_294_; uint64_t v___x_295_; uint64_t v___x_296_; uint64_t v___x_297_; uint64_t v___x_298_; uint64_t v___x_299_; 
v___x_293_ = lean_uint64_mix_hash(v___x_290_, v___y_292_);
v___x_294_ = lp_mathlib_Mathlib_CrossRef_instHashableDatabase_hash(v_database_287_);
v___x_295_ = lean_uint64_mix_hash(v___x_293_, v___x_294_);
v___x_296_ = lean_string_hash(v_tag_288_);
v___x_297_ = lean_uint64_mix_hash(v___x_295_, v___x_296_);
v___x_298_ = lean_string_hash(v_comment_289_);
v___x_299_ = lean_uint64_mix_hash(v___x_297_, v___x_298_);
return v___x_299_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_instHashableTag_hash___boxed(lean_object* v_x_302_){
_start:
{
uint64_t v_res_303_; lean_object* v_r_304_; 
v_res_303_ = lp_mathlib_Mathlib_CrossRef_instHashableTag_hash(v_x_302_);
lean_dec_ref(v_x_302_);
v_r_304_ = lean_box_uint64(v_res_303_);
return v_r_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(lean_object* v_tags_307_, lean_object* v_x_308_){
_start:
{
lean_inc_ref(v_tags_307_);
return v_tags_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2____boxed(lean_object* v_tags_309_, lean_object* v_x_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(v_tags_309_, v_x_310_);
lean_dec_ref(v_x_310_);
lean_dec_ref(v_tags_309_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(lean_object* v_tags_312_){
_start:
{
lean_inc_ref(v_tags_312_);
return v_tags_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2____boxed(lean_object* v_tags_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(v_tags_313_);
lean_dec_ref(v_tags_313_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(lean_object* v_es_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lean_array_mk(v_es_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_));
v___x_336_ = l_Lean_registerSimplePersistentEnvExtension___redArg(v___x_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2____boxed(lean_object* v_a_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_();
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___redArg___lam__0(lean_object* v_declName_339_, lean_object* v_db_340_, lean_object* v_tag_341_, lean_object* v_comment_342_, lean_object* v_x_343_){
_start:
{
lean_object* v___x_344_; lean_object* v_toEnvExtension_345_; lean_object* v_asyncMode_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_344_ = lp_mathlib_Mathlib_CrossRef_tagExt;
v_toEnvExtension_345_ = lean_ctor_get(v___x_344_, 0);
v_asyncMode_346_ = lean_ctor_get(v_toEnvExtension_345_, 2);
v___x_347_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_347_, 0, v_declName_339_);
lean_ctor_set(v___x_347_, 1, v_db_340_);
lean_ctor_set(v___x_347_, 2, v_tag_341_);
lean_ctor_set(v___x_347_, 3, v_comment_342_);
v___x_348_ = lean_box(0);
v___x_349_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_344_, v_x_343_, v___x_347_, v_asyncMode_346_, v___x_348_);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___redArg(lean_object* v_inst_350_, lean_object* v_declName_351_, lean_object* v_db_352_, lean_object* v_tag_353_, lean_object* v_comment_354_){
_start:
{
lean_object* v_modifyEnv_355_; lean_object* v___f_356_; lean_object* v___x_357_; 
v_modifyEnv_355_ = lean_ctor_get(v_inst_350_, 1);
lean_inc(v_modifyEnv_355_);
lean_dec_ref(v_inst_350_);
v___f_356_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CrossRef_addTagEntry___redArg___lam__0), 5, 4);
lean_closure_set(v___f_356_, 0, v_declName_351_);
lean_closure_set(v___f_356_, 1, v_db_352_);
lean_closure_set(v___f_356_, 2, v_tag_353_);
lean_closure_set(v___f_356_, 3, v_comment_354_);
v___x_357_ = lean_apply_1(v_modifyEnv_355_, v___f_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry(lean_object* v_m_358_, lean_object* v_inst_359_, lean_object* v_declName_360_, lean_object* v_db_361_, lean_object* v_tag_362_, lean_object* v_comment_363_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_mathlib_Mathlib_CrossRef_addTagEntry___redArg(v_inst_359_, v_declName_360_, v_db_361_, v_tag_362_, v_comment_363_);
return v___x_364_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_365_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_366_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__0, &lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__0);
v___x_367_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_367_, 0, v___x_366_);
return v___x_367_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_368_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__1, &lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__1);
v___x_369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
lean_ctor_set(v___x_369_, 1, v___x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg(lean_object* v_declName_370_, lean_object* v_db_371_, lean_object* v_tag_372_, lean_object* v_comment_373_, lean_object* v___y_374_){
_start:
{
lean_object* v___x_376_; lean_object* v_env_377_; lean_object* v_nextMacroScope_378_; lean_object* v_ngen_379_; lean_object* v_auxDeclNGen_380_; lean_object* v_traceState_381_; lean_object* v_messages_382_; lean_object* v_infoState_383_; lean_object* v_snapshotTasks_384_; lean_object* v___x_386_; uint8_t v_isShared_387_; uint8_t v_isSharedCheck_401_; 
v___x_376_ = lean_st_ref_take(v___y_374_);
v_env_377_ = lean_ctor_get(v___x_376_, 0);
v_nextMacroScope_378_ = lean_ctor_get(v___x_376_, 1);
v_ngen_379_ = lean_ctor_get(v___x_376_, 2);
v_auxDeclNGen_380_ = lean_ctor_get(v___x_376_, 3);
v_traceState_381_ = lean_ctor_get(v___x_376_, 4);
v_messages_382_ = lean_ctor_get(v___x_376_, 6);
v_infoState_383_ = lean_ctor_get(v___x_376_, 7);
v_snapshotTasks_384_ = lean_ctor_get(v___x_376_, 8);
v_isSharedCheck_401_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_401_ == 0)
{
lean_object* v_unused_402_; 
v_unused_402_ = lean_ctor_get(v___x_376_, 5);
lean_dec(v_unused_402_);
v___x_386_ = v___x_376_;
v_isShared_387_ = v_isSharedCheck_401_;
goto v_resetjp_385_;
}
else
{
lean_inc(v_snapshotTasks_384_);
lean_inc(v_infoState_383_);
lean_inc(v_messages_382_);
lean_inc(v_traceState_381_);
lean_inc(v_auxDeclNGen_380_);
lean_inc(v_ngen_379_);
lean_inc(v_nextMacroScope_378_);
lean_inc(v_env_377_);
lean_dec(v___x_376_);
v___x_386_ = lean_box(0);
v_isShared_387_ = v_isSharedCheck_401_;
goto v_resetjp_385_;
}
v_resetjp_385_:
{
lean_object* v___x_388_; lean_object* v_toEnvExtension_389_; lean_object* v_asyncMode_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_396_; 
v___x_388_ = lp_mathlib_Mathlib_CrossRef_tagExt;
v_toEnvExtension_389_ = lean_ctor_get(v___x_388_, 0);
v_asyncMode_390_ = lean_ctor_get(v_toEnvExtension_389_, 2);
v___x_391_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_391_, 0, v_declName_370_);
lean_ctor_set(v___x_391_, 1, v_db_371_);
lean_ctor_set(v___x_391_, 2, v_tag_372_);
lean_ctor_set(v___x_391_, 3, v_comment_373_);
v___x_392_ = lean_box(0);
v___x_393_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_388_, v_env_377_, v___x_391_, v_asyncMode_390_, v___x_392_);
v___x_394_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2, &lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2_once, _init_lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2);
if (v_isShared_387_ == 0)
{
lean_ctor_set(v___x_386_, 5, v___x_394_);
lean_ctor_set(v___x_386_, 0, v___x_393_);
v___x_396_ = v___x_386_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v___x_393_);
lean_ctor_set(v_reuseFailAlloc_400_, 1, v_nextMacroScope_378_);
lean_ctor_set(v_reuseFailAlloc_400_, 2, v_ngen_379_);
lean_ctor_set(v_reuseFailAlloc_400_, 3, v_auxDeclNGen_380_);
lean_ctor_set(v_reuseFailAlloc_400_, 4, v_traceState_381_);
lean_ctor_set(v_reuseFailAlloc_400_, 5, v___x_394_);
lean_ctor_set(v_reuseFailAlloc_400_, 6, v_messages_382_);
lean_ctor_set(v_reuseFailAlloc_400_, 7, v_infoState_383_);
lean_ctor_set(v_reuseFailAlloc_400_, 8, v_snapshotTasks_384_);
v___x_396_ = v_reuseFailAlloc_400_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_397_ = lean_st_ref_set(v___y_374_, v___x_396_);
v___x_398_ = lean_box(0);
v___x_399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_399_, 0, v___x_398_);
return v___x_399_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___boxed(lean_object* v_declName_403_, lean_object* v_db_404_, lean_object* v_tag_405_, lean_object* v_comment_406_, lean_object* v___y_407_, lean_object* v___y_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg(v_declName_403_, v_db_404_, v_tag_405_, v_comment_406_, v___y_407_);
lean_dec(v___y_407_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2(lean_object* v_declName_410_, lean_object* v_db_411_, lean_object* v_tag_412_, lean_object* v_comment_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg(v_declName_410_, v_db_411_, v_tag_412_, v_comment_413_, v___y_415_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___boxed(lean_object* v_declName_418_, lean_object* v_db_419_, lean_object* v_tag_420_, lean_object* v_comment_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2(v_declName_418_, v_db_419_, v_tag_420_, v_comment_421_, v___y_422_, v___y_423_);
lean_dec(v___y_423_);
lean_dec_ref(v___y_422_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Mathlib_CrossRef_addCrossRefDoc_spec__0(lean_object* v_a_426_, lean_object* v_a_427_){
_start:
{
if (lean_obj_tag(v_a_426_) == 0)
{
lean_object* v___x_428_; 
v___x_428_ = l_List_reverse___redArg(v_a_427_);
return v___x_428_;
}
else
{
lean_object* v_head_429_; lean_object* v_tail_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_441_; 
v_head_429_ = lean_ctor_get(v_a_426_, 0);
v_tail_430_ = lean_ctor_get(v_a_426_, 1);
v_isSharedCheck_441_ = !lean_is_exclusive(v_a_426_);
if (v_isSharedCheck_441_ == 0)
{
v___x_432_ = v_a_426_;
v_isShared_433_ = v_isSharedCheck_441_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_tail_430_);
lean_inc(v_head_429_);
lean_dec(v_a_426_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_441_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_434_; uint8_t v___x_435_; 
v___x_434_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___x_435_ = lean_string_dec_eq(v_head_429_, v___x_434_);
if (v___x_435_ == 0)
{
lean_object* v___x_437_; 
if (v_isShared_433_ == 0)
{
lean_ctor_set(v___x_432_, 1, v_a_427_);
v___x_437_ = v___x_432_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v_head_429_);
lean_ctor_set(v_reuseFailAlloc_439_, 1, v_a_427_);
v___x_437_ = v_reuseFailAlloc_439_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
v_a_426_ = v_tail_430_;
v_a_427_ = v___x_437_;
goto _start;
}
}
else
{
lean_del_object(v___x_432_);
lean_dec(v_head_429_);
v_a_426_ = v_tail_430_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_442_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1(void){
_start:
{
lean_object* v___x_443_; lean_object* v___x_444_; 
v___x_443_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__0);
v___x_444_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_444_, 0, v___x_443_);
return v___x_444_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2(void){
_start:
{
lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; 
v___x_445_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1);
v___x_446_ = lean_unsigned_to_nat(0u);
v___x_447_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_447_, 0, v___x_446_);
lean_ctor_set(v___x_447_, 1, v___x_446_);
lean_ctor_set(v___x_447_, 2, v___x_446_);
lean_ctor_set(v___x_447_, 3, v___x_446_);
lean_ctor_set(v___x_447_, 4, v___x_445_);
lean_ctor_set(v___x_447_, 5, v___x_445_);
lean_ctor_set(v___x_447_, 6, v___x_445_);
lean_ctor_set(v___x_447_, 7, v___x_445_);
lean_ctor_set(v___x_447_, 8, v___x_445_);
lean_ctor_set(v___x_447_, 9, v___x_445_);
return v___x_447_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__3(void){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; 
v___x_448_ = lean_unsigned_to_nat(32u);
v___x_449_ = lean_mk_empty_array_with_capacity(v___x_448_);
v___x_450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_450_, 0, v___x_449_);
return v___x_450_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__4(void){
_start:
{
size_t v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_451_ = ((size_t)5ULL);
v___x_452_ = lean_unsigned_to_nat(0u);
v___x_453_ = lean_unsigned_to_nat(32u);
v___x_454_ = lean_mk_empty_array_with_capacity(v___x_453_);
v___x_455_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__3);
v___x_456_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_456_, 0, v___x_455_);
lean_ctor_set(v___x_456_, 1, v___x_454_);
lean_ctor_set(v___x_456_, 2, v___x_452_);
lean_ctor_set(v___x_456_, 3, v___x_452_);
lean_ctor_set_usize(v___x_456_, 4, v___x_451_);
return v___x_456_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5(void){
_start:
{
lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; 
v___x_457_ = lean_box(1);
v___x_458_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__4);
v___x_459_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__1);
v___x_460_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_460_, 0, v___x_459_);
lean_ctor_set(v___x_460_, 1, v___x_458_);
lean_ctor_set(v___x_460_, 2, v___x_457_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3(lean_object* v_msgData_461_, lean_object* v___y_462_, lean_object* v___y_463_){
_start:
{
lean_object* v___x_465_; lean_object* v_env_466_; lean_object* v_options_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_465_ = lean_st_ref_get(v___y_463_);
v_env_466_ = lean_ctor_get(v___x_465_, 0);
lean_inc_ref(v_env_466_);
lean_dec(v___x_465_);
v_options_467_ = lean_ctor_get(v___y_462_, 2);
v___x_468_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2);
v___x_469_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5);
lean_inc_ref(v_options_467_);
v___x_470_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_470_, 0, v_env_466_);
lean_ctor_set(v___x_470_, 1, v___x_468_);
lean_ctor_set(v___x_470_, 2, v___x_469_);
lean_ctor_set(v___x_470_, 3, v_options_467_);
v___x_471_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_471_, 0, v___x_470_);
lean_ctor_set(v___x_471_, 1, v_msgData_461_);
v___x_472_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_472_, 0, v___x_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___boxed(lean_object* v_msgData_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3(v_msgData_473_, v___y_474_, v___y_475_);
lean_dec(v___y_475_);
lean_dec_ref(v___y_474_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(lean_object* v_msg_478_, lean_object* v___y_479_, lean_object* v___y_480_){
_start:
{
lean_object* v_ref_482_; lean_object* v___x_483_; lean_object* v_a_484_; lean_object* v___x_486_; uint8_t v_isShared_487_; uint8_t v_isSharedCheck_492_; 
v_ref_482_ = lean_ctor_get(v___y_479_, 5);
v___x_483_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3(v_msg_478_, v___y_479_, v___y_480_);
v_a_484_ = lean_ctor_get(v___x_483_, 0);
v_isSharedCheck_492_ = !lean_is_exclusive(v___x_483_);
if (v_isSharedCheck_492_ == 0)
{
v___x_486_ = v___x_483_;
v_isShared_487_ = v_isSharedCheck_492_;
goto v_resetjp_485_;
}
else
{
lean_inc(v_a_484_);
lean_dec(v___x_483_);
v___x_486_ = lean_box(0);
v_isShared_487_ = v_isSharedCheck_492_;
goto v_resetjp_485_;
}
v_resetjp_485_:
{
lean_object* v___x_488_; lean_object* v___x_490_; 
lean_inc(v_ref_482_);
v___x_488_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_488_, 0, v_ref_482_);
lean_ctor_set(v___x_488_, 1, v_a_484_);
if (v_isShared_487_ == 0)
{
lean_ctor_set_tag(v___x_486_, 1);
lean_ctor_set(v___x_486_, 0, v___x_488_);
v___x_490_ = v___x_486_;
goto v_reusejp_489_;
}
else
{
lean_object* v_reuseFailAlloc_491_; 
v_reuseFailAlloc_491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_491_, 0, v___x_488_);
v___x_490_ = v_reuseFailAlloc_491_;
goto v_reusejp_489_;
}
v_reusejp_489_:
{
return v___x_490_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg___boxed(lean_object* v_msg_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v_msg_493_, v___y_494_, v___y_495_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
return v_res_497_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__1(void){
_start:
{
lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_499_ = ((lean_object*)(lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__0));
v___x_500_ = l_Lean_stringToMessageData(v___x_499_);
return v___x_500_;
}
}
static lean_object* _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__3(void){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; 
v___x_502_ = ((lean_object*)(lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__2));
v___x_503_ = l_Lean_stringToMessageData(v___x_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1(lean_object* v_declName_504_, lean_object* v_docString_505_, lean_object* v___y_506_, lean_object* v___y_507_){
_start:
{
lean_object* v___y_510_; lean_object* v___x_535_; lean_object* v_env_536_; lean_object* v___x_537_; 
v___x_535_ = lean_st_ref_get(v___y_507_);
v_env_536_ = lean_ctor_get(v___x_535_, 0);
lean_inc_ref(v_env_536_);
lean_dec(v___x_535_);
v___x_537_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_536_, v_declName_504_);
lean_dec_ref(v_env_536_);
if (lean_obj_tag(v___x_537_) == 0)
{
v___y_510_ = v___y_507_;
goto v___jp_509_;
}
else
{
uint8_t v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; 
lean_dec_ref_known(v___x_537_, 1);
lean_dec_ref(v_docString_505_);
v___x_538_ = 0;
v___x_539_ = lean_obj_once(&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__1, &lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__1_once, _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__1);
v___x_540_ = l_Lean_MessageData_ofConstName(v_declName_504_, v___x_538_);
v___x_541_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_541_, 0, v___x_539_);
lean_ctor_set(v___x_541_, 1, v___x_540_);
v___x_542_ = lean_obj_once(&lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__3, &lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__3_once, _init_lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___closed__3);
v___x_543_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_543_, 0, v___x_541_);
lean_ctor_set(v___x_543_, 1, v___x_542_);
v___x_544_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v___x_543_, v___y_506_, v___y_507_);
return v___x_544_;
}
v___jp_509_:
{
lean_object* v___x_511_; lean_object* v_env_512_; lean_object* v_nextMacroScope_513_; lean_object* v_ngen_514_; lean_object* v_auxDeclNGen_515_; lean_object* v_traceState_516_; lean_object* v_messages_517_; lean_object* v_infoState_518_; lean_object* v_snapshotTasks_519_; lean_object* v___x_521_; uint8_t v_isShared_522_; uint8_t v_isSharedCheck_533_; 
v___x_511_ = lean_st_ref_take(v___y_510_);
v_env_512_ = lean_ctor_get(v___x_511_, 0);
v_nextMacroScope_513_ = lean_ctor_get(v___x_511_, 1);
v_ngen_514_ = lean_ctor_get(v___x_511_, 2);
v_auxDeclNGen_515_ = lean_ctor_get(v___x_511_, 3);
v_traceState_516_ = lean_ctor_get(v___x_511_, 4);
v_messages_517_ = lean_ctor_get(v___x_511_, 6);
v_infoState_518_ = lean_ctor_get(v___x_511_, 7);
v_snapshotTasks_519_ = lean_ctor_get(v___x_511_, 8);
v_isSharedCheck_533_ = !lean_is_exclusive(v___x_511_);
if (v_isSharedCheck_533_ == 0)
{
lean_object* v_unused_534_; 
v_unused_534_ = lean_ctor_get(v___x_511_, 5);
lean_dec(v_unused_534_);
v___x_521_ = v___x_511_;
v_isShared_522_ = v_isSharedCheck_533_;
goto v_resetjp_520_;
}
else
{
lean_inc(v_snapshotTasks_519_);
lean_inc(v_infoState_518_);
lean_inc(v_messages_517_);
lean_inc(v_traceState_516_);
lean_inc(v_auxDeclNGen_515_);
lean_inc(v_ngen_514_);
lean_inc(v_nextMacroScope_513_);
lean_inc(v_env_512_);
lean_dec(v___x_511_);
v___x_521_ = lean_box(0);
v_isShared_522_ = v_isSharedCheck_533_;
goto v_resetjp_520_;
}
v_resetjp_520_:
{
lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_528_; 
v___x_523_ = l_Lean_docStringExt;
v___x_524_ = l_String_removeLeadingSpaces(v_docString_505_);
v___x_525_ = l_Lean_MapDeclarationExtension_insert___redArg(v___x_523_, v_env_512_, v_declName_504_, v___x_524_);
v___x_526_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2, &lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2_once, _init_lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg___closed__2);
if (v_isShared_522_ == 0)
{
lean_ctor_set(v___x_521_, 5, v___x_526_);
lean_ctor_set(v___x_521_, 0, v___x_525_);
v___x_528_ = v___x_521_;
goto v_reusejp_527_;
}
else
{
lean_object* v_reuseFailAlloc_532_; 
v_reuseFailAlloc_532_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_532_, 0, v___x_525_);
lean_ctor_set(v_reuseFailAlloc_532_, 1, v_nextMacroScope_513_);
lean_ctor_set(v_reuseFailAlloc_532_, 2, v_ngen_514_);
lean_ctor_set(v_reuseFailAlloc_532_, 3, v_auxDeclNGen_515_);
lean_ctor_set(v_reuseFailAlloc_532_, 4, v_traceState_516_);
lean_ctor_set(v_reuseFailAlloc_532_, 5, v___x_526_);
lean_ctor_set(v_reuseFailAlloc_532_, 6, v_messages_517_);
lean_ctor_set(v_reuseFailAlloc_532_, 7, v_infoState_518_);
lean_ctor_set(v_reuseFailAlloc_532_, 8, v_snapshotTasks_519_);
v___x_528_ = v_reuseFailAlloc_532_;
goto v_reusejp_527_;
}
v_reusejp_527_:
{
lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; 
v___x_529_ = lean_st_ref_set(v___y_510_, v___x_528_);
v___x_530_ = lean_box(0);
v___x_531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_531_, 0, v___x_530_);
return v___x_531_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1___boxed(lean_object* v_declName_545_, lean_object* v_docString_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_){
_start:
{
lean_object* v_res_550_; 
v_res_550_ = lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1(v_declName_545_, v_docString_546_, v___y_547_, v___y_548_);
lean_dec(v___y_548_);
lean_dec_ref(v___y_547_);
return v_res_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(lean_object* v_db_557_, lean_object* v_decl_558_, lean_object* v_idStr_559_, lean_object* v_comment_560_, lean_object* v_a_561_, lean_object* v_a_562_){
_start:
{
lean_object* v___x_564_; lean_object* v_env_565_; uint8_t v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_564_ = lean_st_ref_get(v_a_562_);
v_env_565_ = lean_ctor_get(v___x_564_, 0);
lean_inc_ref(v_env_565_);
lean_dec(v___x_564_);
v___x_566_ = 1;
v___x_567_ = l_Lean_Options_empty;
v___x_568_ = lean_box(0);
v___x_569_ = lean_box(0);
lean_inc(v_decl_558_);
v___x_570_ = l_Lean_findDocString_x3f(v_env_565_, v_decl_558_, v___x_566_, v___x_567_, v___x_568_, v___x_569_);
if (lean_obj_tag(v___x_570_) == 0)
{
lean_object* v_a_571_; lean_object* v___y_573_; lean_object* v___y_574_; lean_object* v___y_596_; 
v_a_571_ = lean_ctor_get(v___x_570_, 0);
lean_inc(v_a_571_);
lean_dec_ref_known(v___x_570_, 1);
if (lean_obj_tag(v_a_571_) == 0)
{
lean_object* v___x_605_; 
v___x_605_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___y_596_ = v___x_605_;
goto v___jp_595_;
}
else
{
lean_object* v_val_606_; 
v_val_606_ = lean_ctor_get(v_a_571_, 0);
lean_inc(v_val_606_);
lean_dec_ref_known(v_a_571_, 1);
v___y_596_ = v_val_606_;
goto v___jp_595_;
}
v___jp_572_:
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
v___x_575_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__0));
v___x_576_ = lp_mathlib_Mathlib_CrossRef_Database_label(v_db_557_);
v___x_577_ = lean_string_append(v___x_575_, v___x_576_);
lean_dec_ref(v___x_576_);
v___x_578_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__1));
v___x_579_ = lean_string_append(v___x_577_, v___x_578_);
v___x_580_ = lean_string_append(v___x_579_, v_idStr_559_);
v___x_581_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__2));
v___x_582_ = lean_string_append(v___x_580_, v___x_581_);
lean_inc_ref(v_idStr_559_);
v___x_583_ = lp_mathlib_Mathlib_CrossRef_Database_url(v_db_557_, v_idStr_559_);
v___x_584_ = lean_string_append(v___x_582_, v___x_583_);
lean_dec_ref(v___x_583_);
v___x_585_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__3));
v___x_586_ = lean_string_append(v___x_584_, v___x_585_);
v___x_587_ = lean_string_append(v___x_586_, v___y_574_);
lean_dec_ref(v___y_574_);
v___x_588_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__4));
v___x_589_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_589_, 0, v___x_587_);
lean_ctor_set(v___x_589_, 1, v___x_569_);
v___x_590_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_590_, 0, v___y_573_);
lean_ctor_set(v___x_590_, 1, v___x_589_);
v___x_591_ = lp_mathlib_List_filterTR_loop___at___00Mathlib_CrossRef_addCrossRefDoc_spec__0(v___x_590_, v___x_569_);
v___x_592_ = l_String_intercalate(v___x_588_, v___x_591_);
lean_inc(v_decl_558_);
v___x_593_ = lp_mathlib_Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1(v_decl_558_, v___x_592_, v_a_561_, v_a_562_);
if (lean_obj_tag(v___x_593_) == 0)
{
lean_object* v___x_594_; 
lean_dec_ref_known(v___x_593_, 1);
v___x_594_ = lp_mathlib_Mathlib_CrossRef_addTagEntry___at___00Mathlib_CrossRef_addCrossRefDoc_spec__2___redArg(v_decl_558_, v_db_557_, v_idStr_559_, v_comment_560_, v_a_562_);
return v___x_594_;
}
else
{
lean_dec_ref(v_comment_560_);
lean_dec_ref(v_idStr_559_);
lean_dec(v_decl_558_);
lean_dec(v_db_557_);
return v___x_593_;
}
}
v___jp_595_:
{
lean_object* v___x_597_; lean_object* v___x_598_; uint8_t v___x_599_; 
v___x_597_ = lean_string_utf8_byte_size(v_comment_560_);
v___x_598_ = lean_unsigned_to_nat(0u);
v___x_599_ = lean_nat_dec_eq(v___x_597_, v___x_598_);
if (v___x_599_ == 0)
{
lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; 
v___x_600_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__5));
v___x_601_ = lean_string_append(v___x_600_, v_comment_560_);
v___x_602_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__3));
v___x_603_ = lean_string_append(v___x_601_, v___x_602_);
v___y_573_ = v___y_596_;
v___y_574_ = v___x_603_;
goto v___jp_572_;
}
else
{
lean_object* v___x_604_; 
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___y_573_ = v___y_596_;
v___y_574_ = v___x_604_;
goto v___jp_572_;
}
}
}
else
{
lean_object* v_a_607_; lean_object* v___x_609_; uint8_t v_isShared_610_; uint8_t v_isSharedCheck_619_; 
lean_dec_ref(v_comment_560_);
lean_dec_ref(v_idStr_559_);
lean_dec(v_decl_558_);
lean_dec(v_db_557_);
v_a_607_ = lean_ctor_get(v___x_570_, 0);
v_isSharedCheck_619_ = !lean_is_exclusive(v___x_570_);
if (v_isSharedCheck_619_ == 0)
{
v___x_609_ = v___x_570_;
v_isShared_610_ = v_isSharedCheck_619_;
goto v_resetjp_608_;
}
else
{
lean_inc(v_a_607_);
lean_dec(v___x_570_);
v___x_609_ = lean_box(0);
v_isShared_610_ = v_isSharedCheck_619_;
goto v_resetjp_608_;
}
v_resetjp_608_:
{
lean_object* v_ref_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_617_; 
v_ref_611_ = lean_ctor_get(v_a_561_, 5);
v___x_612_ = lean_io_error_to_string(v_a_607_);
v___x_613_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_613_, 0, v___x_612_);
v___x_614_ = l_Lean_MessageData_ofFormat(v___x_613_);
lean_inc(v_ref_611_);
v___x_615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_615_, 0, v_ref_611_);
lean_ctor_set(v___x_615_, 1, v___x_614_);
if (v_isShared_610_ == 0)
{
lean_ctor_set(v___x_609_, 0, v___x_615_);
v___x_617_ = v___x_609_;
goto v_reusejp_616_;
}
else
{
lean_object* v_reuseFailAlloc_618_; 
v_reuseFailAlloc_618_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_618_, 0, v___x_615_);
v___x_617_ = v_reuseFailAlloc_618_;
goto v_reusejp_616_;
}
v_reusejp_616_:
{
return v___x_617_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___boxed(lean_object* v_db_620_, lean_object* v_decl_621_, lean_object* v_idStr_622_, lean_object* v_comment_623_, lean_object* v_a_624_, lean_object* v_a_625_, lean_object* v_a_626_){
_start:
{
lean_object* v_res_627_; 
v_res_627_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v_db_620_, v_decl_621_, v_idStr_622_, v_comment_623_, v_a_624_, v_a_625_);
lean_dec(v_a_625_);
lean_dec_ref(v_a_624_);
return v_res_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1(lean_object* v_00_u03b1_628_, lean_object* v_msg_629_, lean_object* v___y_630_, lean_object* v___y_631_){
_start:
{
lean_object* v___x_633_; 
v___x_633_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v_msg_629_, v___y_630_, v___y_631_);
return v___x_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___boxed(lean_object* v_00_u03b1_634_, lean_object* v_msg_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1(v_00_u03b1_634_, v_msg_635_, v___y_636_, v___y_637_);
lean_dec(v___y_637_);
lean_dec_ref(v___y_636_);
return v_res_639_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0(lean_object* v_x_644_, lean_object* v_x_645_){
_start:
{
if (lean_obj_tag(v_x_644_) == 0)
{
if (lean_obj_tag(v_x_645_) == 0)
{
uint8_t v___x_646_; 
v___x_646_ = 1;
return v___x_646_;
}
else
{
uint8_t v___x_647_; 
v___x_647_ = 0;
return v___x_647_;
}
}
else
{
if (lean_obj_tag(v_x_645_) == 0)
{
uint8_t v___x_648_; 
v___x_648_ = 0;
return v___x_648_;
}
else
{
lean_object* v_val_649_; lean_object* v_val_650_; uint8_t v___x_651_; 
v_val_649_ = lean_ctor_get(v_x_644_, 0);
v_val_650_ = lean_ctor_get(v_x_645_, 0);
v___x_651_ = l_Lean_Parser_instBEqError_beq(v_val_649_, v_val_650_);
return v___x_651_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0___boxed(lean_object* v_x_652_, lean_object* v_x_653_){
_start:
{
uint8_t v_res_654_; lean_object* v_r_655_; 
v_res_654_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0(v_x_652_, v_x_653_);
lean_dec(v_x_653_);
lean_dec(v_x_652_);
v_r_655_ = lean_box(v_res_654_);
return v_r_655_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_stacksTagFn___lam__0(uint32_t v_c_656_){
_start:
{
uint8_t v___y_658_; uint32_t v___x_668_; uint8_t v___x_669_; 
v___x_668_ = 65;
v___x_669_ = lean_uint32_dec_le(v___x_668_, v_c_656_);
if (v___x_669_ == 0)
{
goto v___jp_663_;
}
else
{
uint32_t v___x_670_; uint8_t v___x_671_; 
v___x_670_ = 90;
v___x_671_ = lean_uint32_dec_le(v_c_656_, v___x_670_);
if (v___x_671_ == 0)
{
goto v___jp_663_;
}
else
{
return v___x_671_;
}
}
v___jp_657_:
{
if (v___y_658_ == 0)
{
uint32_t v___x_659_; uint8_t v___x_660_; 
v___x_659_ = 48;
v___x_660_ = lean_uint32_dec_le(v___x_659_, v_c_656_);
if (v___x_660_ == 0)
{
return v___x_660_;
}
else
{
uint32_t v___x_661_; uint8_t v___x_662_; 
v___x_661_ = 57;
v___x_662_ = lean_uint32_dec_le(v_c_656_, v___x_661_);
return v___x_662_;
}
}
else
{
return v___y_658_;
}
}
v___jp_663_:
{
uint32_t v___x_664_; uint8_t v___x_665_; 
v___x_664_ = 97;
v___x_665_ = lean_uint32_dec_le(v___x_664_, v_c_656_);
if (v___x_665_ == 0)
{
v___y_658_ = v___x_665_;
goto v___jp_657_;
}
else
{
uint32_t v___x_666_; uint8_t v___x_667_; 
v___x_666_ = 122;
v___x_667_ = lean_uint32_dec_le(v_c_656_, v___x_666_);
v___y_658_ = v___x_667_;
goto v___jp_657_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagFn___lam__0___boxed(lean_object* v_c_672_){
_start:
{
uint32_t v_c_boxed_673_; uint8_t v_res_674_; lean_object* v_r_675_; 
v_c_boxed_673_ = lean_unbox_uint32(v_c_672_);
lean_dec(v_c_672_);
v_res_674_ = lp_mathlib_Mathlib_CrossRef_stacksTagFn___lam__0(v_c_boxed_673_);
v_r_675_ = lean_box(v_res_674_);
return v_r_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_CrossRef_stacksTagFn_spec__1(uint8_t v___x_676_, lean_object* v_s_677_, lean_object* v_pos_678_){
_start:
{
lean_object* v_str_679_; lean_object* v_startInclusive_680_; lean_object* v_endExclusive_681_; lean_object* v___x_682_; uint8_t v___y_684_; lean_object* v___x_690_; lean_object* v___x_691_; uint8_t v___x_692_; 
v_str_679_ = lean_ctor_get(v_s_677_, 0);
v_startInclusive_680_ = lean_ctor_get(v_s_677_, 1);
v_endExclusive_681_ = lean_ctor_get(v_s_677_, 2);
v___x_682_ = lean_nat_add(v_startInclusive_680_, v_pos_678_);
v___x_690_ = lean_unsigned_to_nat(0u);
v___x_691_ = lean_nat_sub(v_endExclusive_681_, v___x_682_);
v___x_692_ = lean_nat_dec_eq(v___x_690_, v___x_691_);
lean_dec(v___x_691_);
if (v___x_692_ == 0)
{
uint32_t v___x_693_; uint8_t v___y_695_; uint32_t v___x_700_; uint8_t v___x_701_; 
v___x_693_ = lean_string_utf8_get_fast(v_str_679_, v___x_682_);
v___x_700_ = 48;
v___x_701_ = lean_uint32_dec_le(v___x_700_, v___x_693_);
if (v___x_701_ == 0)
{
v___y_695_ = v___x_701_;
goto v___jp_694_;
}
else
{
uint32_t v___x_702_; uint8_t v___x_703_; 
v___x_702_ = 57;
v___x_703_ = lean_uint32_dec_le(v___x_693_, v___x_702_);
v___y_695_ = v___x_703_;
goto v___jp_694_;
}
v___jp_694_:
{
if (v___y_695_ == 0)
{
uint32_t v___x_696_; uint8_t v___x_697_; 
v___x_696_ = 65;
v___x_697_ = lean_uint32_dec_le(v___x_696_, v___x_693_);
if (v___x_697_ == 0)
{
lean_dec(v___x_682_);
return v_pos_678_;
}
else
{
uint32_t v___x_698_; uint8_t v___x_699_; 
v___x_698_ = 90;
v___x_699_ = lean_uint32_dec_le(v___x_693_, v___x_698_);
if (v___x_699_ == 0)
{
lean_dec(v___x_682_);
return v_pos_678_;
}
else
{
v___y_684_ = v___x_676_;
goto v___jp_683_;
}
}
}
else
{
v___y_684_ = v___x_676_;
goto v___jp_683_;
}
}
}
else
{
lean_dec(v___x_682_);
return v_pos_678_;
}
v___jp_683_:
{
if (v___y_684_ == 0)
{
lean_dec(v___x_682_);
return v_pos_678_;
}
else
{
lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; uint8_t v___x_688_; 
v___x_685_ = lean_string_utf8_next_fast(v_str_679_, v___x_682_);
v___x_686_ = lean_nat_sub(v___x_685_, v___x_682_);
lean_dec(v___x_682_);
v___x_687_ = lean_nat_add(v_pos_678_, v___x_686_);
lean_dec(v___x_686_);
v___x_688_ = lean_nat_dec_lt(v_pos_678_, v___x_687_);
if (v___x_688_ == 0)
{
lean_dec(v___x_687_);
return v_pos_678_;
}
else
{
lean_dec(v_pos_678_);
v_pos_678_ = v___x_687_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_CrossRef_stacksTagFn_spec__1___boxed(lean_object* v___x_704_, lean_object* v_s_705_, lean_object* v_pos_706_){
_start:
{
uint8_t v___x_1090__boxed_707_; lean_object* v_res_708_; 
v___x_1090__boxed_707_ = lean_unbox(v___x_704_);
v_res_708_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_CrossRef_stacksTagFn_spec__1(v___x_1090__boxed_707_, v_s_705_, v_pos_706_);
lean_dec_ref(v_s_705_);
return v_res_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagFn(lean_object* v_c_713_, lean_object* v_s_714_){
_start:
{
lean_object* v_pos_715_; lean_object* v___f_716_; lean_object* v_s_717_; lean_object* v_pos_718_; lean_object* v_errorMsg_719_; lean_object* v___x_720_; uint8_t v___x_721_; 
v_pos_715_ = lean_ctor_get(v_s_714_, 2);
lean_inc(v_pos_715_);
v___f_716_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__0));
v_s_717_ = l_Lean_Parser_takeWhileFn(v___f_716_, v_c_713_, v_s_714_);
v_pos_718_ = lean_ctor_get(v_s_717_, 2);
lean_inc(v_pos_718_);
v_errorMsg_719_ = lean_ctor_get(v_s_717_, 4);
lean_inc(v_errorMsg_719_);
v___x_720_ = lean_box(0);
v___x_721_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0(v_errorMsg_719_, v___x_720_);
lean_dec(v_errorMsg_719_);
if (v___x_721_ == 0)
{
lean_dec(v_pos_718_);
lean_dec(v_pos_715_);
lean_dec_ref(v_c_713_);
return v_s_717_;
}
else
{
uint8_t v___x_726_; 
v___x_726_ = lean_nat_dec_eq(v_pos_718_, v_pos_715_);
if (v___x_726_ == 0)
{
lean_object* v_toInputContext_727_; lean_object* v_inputString_728_; lean_object* v_tag_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; uint8_t v___x_734_; 
v_toInputContext_727_ = lean_ctor_get(v_c_713_, 0);
v_inputString_728_ = lean_ctor_get(v_toInputContext_727_, 0);
v_tag_729_ = lean_string_utf8_extract(v_inputString_728_, v_pos_715_, v_pos_718_);
lean_dec(v_pos_718_);
v___x_730_ = lean_unsigned_to_nat(0u);
v___x_731_ = lean_string_utf8_byte_size(v_tag_729_);
lean_inc_ref(v_tag_729_);
v___x_732_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_732_, 0, v_tag_729_);
lean_ctor_set(v___x_732_, 1, v___x_730_);
lean_ctor_set(v___x_732_, 2, v___x_731_);
v___x_733_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00Mathlib_CrossRef_stacksTagFn_spec__1(v___x_721_, v___x_732_, v___x_730_);
lean_dec_ref_known(v___x_732_, 3);
v___x_734_ = lean_nat_dec_eq(v___x_733_, v___x_731_);
lean_dec(v___x_733_);
if (v___x_734_ == 0)
{
lean_dec_ref(v_tag_729_);
lean_dec(v_pos_715_);
lean_dec_ref(v_c_713_);
goto v___jp_722_;
}
else
{
if (v___x_726_ == 0)
{
lean_object* v___x_735_; lean_object* v___x_736_; uint8_t v___x_737_; 
v___x_735_ = lean_string_length(v_tag_729_);
lean_dec_ref(v_tag_729_);
v___x_736_ = lean_unsigned_to_nat(4u);
v___x_737_ = lean_nat_dec_eq(v___x_735_, v___x_736_);
if (v___x_737_ == 0)
{
lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; 
lean_dec(v_pos_715_);
lean_dec_ref(v_c_713_);
v___x_738_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__2));
v___x_739_ = lean_box(0);
v___x_740_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_717_, v___x_738_, v___x_739_, v___x_721_);
return v___x_740_;
}
else
{
lean_object* v___x_741_; lean_object* v___x_742_; 
v___x_741_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1));
v___x_742_ = l_Lean_Parser_mkNodeToken(v___x_741_, v_pos_715_, v___x_721_, v_c_713_, v_s_717_);
return v___x_742_;
}
}
else
{
lean_dec_ref(v_tag_729_);
lean_dec(v_pos_715_);
lean_dec_ref(v_c_713_);
goto v___jp_722_;
}
}
}
else
{
lean_object* v___x_743_; lean_object* v___x_744_; 
lean_dec(v_pos_718_);
lean_dec(v_pos_715_);
lean_dec_ref(v_c_713_);
v___x_743_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__3));
v___x_744_ = l_Lean_Parser_ParserState_mkError(v_s_717_, v___x_743_);
return v___x_744_;
}
}
v___jp_722_:
{
lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_723_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__1));
v___x_724_ = lean_box(0);
v___x_725_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_717_, v___x_723_, v___x_724_, v___x_721_);
return v___x_725_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__0(void){
_start:
{
lean_object* v___x_745_; lean_object* v___x_746_; 
v___x_745_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0));
v___x_746_ = l_Lean_Parser_mkAtomicInfo(v___x_745_);
return v___x_746_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__1(void){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; 
v___x_747_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CrossRef_stacksTagFn), 2, 0);
v___x_748_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__0, &lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__0);
v___x_749_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_749_, 0, v___x_748_);
lean_ctor_set(v___x_749_, 1, v___x_747_);
return v___x_749_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot(void){
_start:
{
lean_object* v___x_750_; 
v___x_750_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__1, &lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot___closed__1);
return v___x_750_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__0(void){
_start:
{
uint8_t v___x_751_; uint8_t v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; 
v___x_751_ = 0;
v___x_752_ = 1;
v___x_753_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1));
v___x_754_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__0));
v___x_755_ = l_Lean_Parser_mkAntiquot(v___x_754_, v___x_753_, v___x_752_, v___x_751_);
return v___x_755_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__1(void){
_start:
{
lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; 
v___x_756_ = lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot;
v___x_757_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__0, &lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__0);
v___x_758_ = l_Lean_Parser_withAntiquot(v___x_757_, v___x_756_);
return v___x_758_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_stacksTagParser(void){
_start:
{
lean_object* v___x_759_; 
v___x_759_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__1, &lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_stacksTagParser___closed__1);
return v___x_759_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_all___at___00Mathlib_CrossRef_wikidataIdFn_spec__0(lean_object* v_x_764_){
_start:
{
if (lean_obj_tag(v_x_764_) == 0)
{
uint8_t v___x_765_; 
v___x_765_ = 1;
return v___x_765_;
}
else
{
lean_object* v_head_766_; lean_object* v_tail_767_; uint8_t v___y_769_; uint32_t v___x_771_; uint32_t v___x_772_; uint8_t v___x_773_; 
v_head_766_ = lean_ctor_get(v_x_764_, 0);
v_tail_767_ = lean_ctor_get(v_x_764_, 1);
v___x_771_ = 48;
v___x_772_ = lean_unbox_uint32(v_head_766_);
v___x_773_ = lean_uint32_dec_le(v___x_771_, v___x_772_);
if (v___x_773_ == 0)
{
v___y_769_ = v___x_773_;
goto v___jp_768_;
}
else
{
uint32_t v___x_774_; uint32_t v___x_775_; uint8_t v___x_776_; 
v___x_774_ = 57;
v___x_775_ = lean_unbox_uint32(v_head_766_);
v___x_776_ = lean_uint32_dec_le(v___x_775_, v___x_774_);
v___y_769_ = v___x_776_;
goto v___jp_768_;
}
v___jp_768_:
{
if (v___y_769_ == 0)
{
return v___y_769_;
}
else
{
v_x_764_ = v_tail_767_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_wikidataIdFn_spec__0___boxed(lean_object* v_x_777_){
_start:
{
uint8_t v_res_778_; lean_object* v_r_779_; 
v_res_778_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_wikidataIdFn_spec__0(v_x_777_);
lean_dec(v_x_777_);
v_r_779_ = lean_box(v_res_778_);
return v_r_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdFn(lean_object* v_c_783_, lean_object* v_s_784_){
_start:
{
lean_object* v_pos_785_; lean_object* v___f_786_; lean_object* v_s_787_; lean_object* v_pos_788_; lean_object* v_errorMsg_789_; lean_object* v___x_790_; uint8_t v___x_791_; 
v_pos_785_ = lean_ctor_get(v_s_784_, 2);
lean_inc(v_pos_785_);
v___f_786_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__0));
v_s_787_ = l_Lean_Parser_takeWhileFn(v___f_786_, v_c_783_, v_s_784_);
v_pos_788_ = lean_ctor_get(v_s_787_, 2);
lean_inc(v_pos_788_);
v_errorMsg_789_ = lean_ctor_get(v_s_787_, 4);
lean_inc(v_errorMsg_789_);
v___x_790_ = lean_box(0);
v___x_791_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0(v_errorMsg_789_, v___x_790_);
lean_dec(v_errorMsg_789_);
if (v___x_791_ == 0)
{
lean_dec(v_pos_788_);
lean_dec(v_pos_785_);
lean_dec_ref(v_c_783_);
return v_s_787_;
}
else
{
uint8_t v___x_796_; 
v___x_796_ = lean_nat_dec_eq(v_pos_788_, v_pos_785_);
if (v___x_796_ == 0)
{
lean_object* v_toInputContext_797_; lean_object* v_inputString_798_; lean_object* v_id_799_; lean_object* v___x_800_; 
v_toInputContext_797_ = lean_ctor_get(v_c_783_, 0);
v_inputString_798_ = lean_ctor_get(v_toInputContext_797_, 0);
v_id_799_ = lean_string_utf8_extract(v_inputString_798_, v_pos_785_, v_pos_788_);
lean_dec(v_pos_788_);
v___x_800_ = lean_string_data(v_id_799_);
if (lean_obj_tag(v___x_800_) == 1)
{
lean_object* v_head_801_; lean_object* v_tail_802_; uint32_t v___x_803_; uint32_t v___x_804_; uint8_t v___x_805_; 
v_head_801_ = lean_ctor_get(v___x_800_, 0);
lean_inc(v_head_801_);
v_tail_802_ = lean_ctor_get(v___x_800_, 1);
lean_inc(v_tail_802_);
lean_dec_ref_known(v___x_800_, 2);
v___x_803_ = 81;
v___x_804_ = lean_unbox_uint32(v_head_801_);
lean_dec(v_head_801_);
v___x_805_ = lean_uint32_dec_eq(v___x_804_, v___x_803_);
if (v___x_805_ == 0)
{
lean_dec(v_tail_802_);
lean_dec(v_pos_785_);
lean_dec_ref(v_c_783_);
goto v___jp_792_;
}
else
{
if (lean_obj_tag(v_tail_802_) == 1)
{
uint8_t v___x_806_; 
v___x_806_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_wikidataIdFn_spec__0(v_tail_802_);
lean_dec_ref_known(v_tail_802_, 2);
if (v___x_806_ == 0)
{
lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; 
lean_dec(v_pos_785_);
lean_dec_ref(v_c_783_);
v___x_807_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__1));
v___x_808_ = lean_box(0);
v___x_809_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_787_, v___x_807_, v___x_808_, v___x_791_);
return v___x_809_;
}
else
{
lean_object* v___x_810_; lean_object* v___x_811_; 
v___x_810_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1));
v___x_811_ = l_Lean_Parser_mkNodeToken(v___x_810_, v_pos_785_, v___x_791_, v_c_783_, v_s_787_);
return v___x_811_;
}
}
else
{
lean_dec(v_tail_802_);
lean_dec(v_pos_785_);
lean_dec_ref(v_c_783_);
goto v___jp_792_;
}
}
}
else
{
lean_dec(v___x_800_);
lean_dec(v_pos_785_);
lean_dec_ref(v_c_783_);
goto v___jp_792_;
}
}
else
{
lean_object* v___x_812_; lean_object* v___x_813_; 
lean_dec(v_pos_788_);
lean_dec(v_pos_785_);
lean_dec_ref(v_c_783_);
v___x_812_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__2));
v___x_813_ = l_Lean_Parser_ParserState_mkError(v_s_787_, v___x_812_);
return v___x_813_;
}
}
v___jp_792_:
{
lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; 
v___x_793_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdFn___closed__0));
v___x_794_ = lean_box(0);
v___x_795_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_787_, v___x_793_, v___x_794_, v___x_791_);
return v___x_795_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__0(void){
_start:
{
lean_object* v___x_814_; lean_object* v___x_815_; 
v___x_814_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__0));
v___x_815_ = l_Lean_Parser_mkAtomicInfo(v___x_814_);
return v___x_815_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__1(void){
_start:
{
lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; 
v___x_816_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CrossRef_wikidataIdFn), 2, 0);
v___x_817_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__0, &lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__0);
v___x_818_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_818_, 0, v___x_817_);
lean_ctor_set(v___x_818_, 1, v___x_816_);
return v___x_818_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot(void){
_start:
{
lean_object* v___x_819_; 
v___x_819_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__1, &lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot___closed__1);
return v___x_819_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__0(void){
_start:
{
uint8_t v___x_820_; uint8_t v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; 
v___x_820_ = 0;
v___x_821_ = 1;
v___x_822_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1));
v___x_823_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__0));
v___x_824_ = l_Lean_Parser_mkAntiquot(v___x_823_, v___x_822_, v___x_821_, v___x_820_);
return v___x_824_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__1(void){
_start:
{
lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; 
v___x_825_ = lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot;
v___x_826_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__0, &lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__0);
v___x_827_ = l_Lean_Parser_withAntiquot(v___x_826_, v___x_825_);
return v___x_827_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_wikidataIdParser(void){
_start:
{
lean_object* v___x_828_; 
v___x_828_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__1, &lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_wikidataIdParser___closed__1);
return v___x_828_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___lam__0(uint32_t v_c_833_){
_start:
{
uint8_t v___y_835_; uint8_t v___y_841_; uint32_t v___x_851_; uint8_t v___x_852_; 
v___x_851_ = 65;
v___x_852_ = lean_uint32_dec_le(v___x_851_, v_c_833_);
if (v___x_852_ == 0)
{
goto v___jp_846_;
}
else
{
uint32_t v___x_853_; uint8_t v___x_854_; 
v___x_853_ = 90;
v___x_854_ = lean_uint32_dec_le(v_c_833_, v___x_853_);
if (v___x_854_ == 0)
{
goto v___jp_846_;
}
else
{
return v___x_854_;
}
}
v___jp_834_:
{
if (v___y_835_ == 0)
{
uint32_t v___x_836_; uint8_t v___x_837_; 
v___x_836_ = 46;
v___x_837_ = lean_uint32_dec_eq(v_c_833_, v___x_836_);
if (v___x_837_ == 0)
{
uint32_t v___x_838_; uint8_t v___x_839_; 
v___x_838_ = 95;
v___x_839_ = lean_uint32_dec_eq(v_c_833_, v___x_838_);
return v___x_839_;
}
else
{
return v___x_837_;
}
}
else
{
return v___y_835_;
}
}
v___jp_840_:
{
if (v___y_841_ == 0)
{
uint32_t v___x_842_; uint8_t v___x_843_; 
v___x_842_ = 48;
v___x_843_ = lean_uint32_dec_le(v___x_842_, v_c_833_);
if (v___x_843_ == 0)
{
v___y_835_ = v___x_843_;
goto v___jp_834_;
}
else
{
uint32_t v___x_844_; uint8_t v___x_845_; 
v___x_844_ = 57;
v___x_845_ = lean_uint32_dec_le(v_c_833_, v___x_844_);
v___y_835_ = v___x_845_;
goto v___jp_834_;
}
}
else
{
return v___y_841_;
}
}
v___jp_846_:
{
uint32_t v___x_847_; uint8_t v___x_848_; 
v___x_847_ = 97;
v___x_848_ = lean_uint32_dec_le(v___x_847_, v_c_833_);
if (v___x_848_ == 0)
{
v___y_841_ = v___x_848_;
goto v___jp_840_;
}
else
{
uint32_t v___x_849_; uint8_t v___x_850_; 
v___x_849_ = 122;
v___x_850_ = lean_uint32_dec_le(v_c_833_, v___x_849_);
v___y_841_ = v___x_850_;
goto v___jp_840_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___lam__0___boxed(lean_object* v_c_855_){
_start:
{
uint32_t v_c_boxed_856_; uint8_t v_res_857_; lean_object* v_r_858_; 
v_c_boxed_856_ = lean_unbox_uint32(v_c_855_);
lean_dec(v_c_855_);
v_res_857_ = lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___lam__0(v_c_boxed_856_);
v_r_858_ = lean_box(v_res_857_);
return v_r_858_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_all___at___00Mathlib_CrossRef_lmfdbIdFn_spec__0(uint8_t v___x_859_, lean_object* v_x_860_){
_start:
{
if (lean_obj_tag(v_x_860_) == 0)
{
uint8_t v___x_861_; 
v___x_861_ = 1;
return v___x_861_;
}
else
{
lean_object* v_head_862_; lean_object* v_tail_863_; uint8_t v___y_865_; uint8_t v___y_868_; uint8_t v___y_876_; uint32_t v___x_883_; uint32_t v___x_884_; uint8_t v___x_885_; 
v_head_862_ = lean_ctor_get(v_x_860_, 0);
v_tail_863_ = lean_ctor_get(v_x_860_, 1);
v___x_883_ = 97;
v___x_884_ = lean_unbox_uint32(v_head_862_);
v___x_885_ = lean_uint32_dec_le(v___x_883_, v___x_884_);
if (v___x_885_ == 0)
{
v___y_876_ = v___x_885_;
goto v___jp_875_;
}
else
{
uint32_t v___x_886_; uint32_t v___x_887_; uint8_t v___x_888_; 
v___x_886_ = 122;
v___x_887_ = lean_unbox_uint32(v_head_862_);
v___x_888_ = lean_uint32_dec_le(v___x_887_, v___x_886_);
v___y_876_ = v___x_888_;
goto v___jp_875_;
}
v___jp_864_:
{
if (v___y_865_ == 0)
{
return v___y_865_;
}
else
{
v_x_860_ = v_tail_863_;
goto _start;
}
}
v___jp_867_:
{
if (v___y_868_ == 0)
{
uint32_t v___x_869_; uint32_t v___x_870_; uint8_t v___x_871_; 
v___x_869_ = 46;
v___x_870_ = lean_unbox_uint32(v_head_862_);
v___x_871_ = lean_uint32_dec_eq(v___x_870_, v___x_869_);
if (v___x_871_ == 0)
{
uint32_t v___x_872_; uint32_t v___x_873_; uint8_t v___x_874_; 
v___x_872_ = 95;
v___x_873_ = lean_unbox_uint32(v_head_862_);
v___x_874_ = lean_uint32_dec_eq(v___x_873_, v___x_872_);
v___y_865_ = v___x_874_;
goto v___jp_864_;
}
else
{
v___y_865_ = v___x_859_;
goto v___jp_864_;
}
}
else
{
v___y_865_ = v___x_859_;
goto v___jp_864_;
}
}
v___jp_875_:
{
if (v___y_876_ == 0)
{
uint32_t v___x_877_; uint32_t v___x_878_; uint8_t v___x_879_; 
v___x_877_ = 48;
v___x_878_ = lean_unbox_uint32(v_head_862_);
v___x_879_ = lean_uint32_dec_le(v___x_877_, v___x_878_);
if (v___x_879_ == 0)
{
v___y_868_ = v___x_879_;
goto v___jp_867_;
}
else
{
uint32_t v___x_880_; uint32_t v___x_881_; uint8_t v___x_882_; 
v___x_880_ = 57;
v___x_881_ = lean_unbox_uint32(v_head_862_);
v___x_882_ = lean_uint32_dec_le(v___x_881_, v___x_880_);
v___y_868_ = v___x_882_;
goto v___jp_867_;
}
}
else
{
v___y_865_ = v___x_859_;
goto v___jp_864_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_lmfdbIdFn_spec__0___boxed(lean_object* v___x_889_, lean_object* v_x_890_){
_start:
{
uint8_t v___x_878__boxed_891_; uint8_t v_res_892_; lean_object* v_r_893_; 
v___x_878__boxed_891_ = lean_unbox(v___x_889_);
v_res_892_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_lmfdbIdFn_spec__0(v___x_878__boxed_891_, v_x_890_);
lean_dec(v_x_890_);
v_r_893_ = lean_box(v_res_892_);
return v_r_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdFn(lean_object* v_c_897_, lean_object* v_s_898_){
_start:
{
lean_object* v_pos_899_; lean_object* v___f_900_; lean_object* v_s_901_; lean_object* v_pos_902_; lean_object* v_errorMsg_903_; lean_object* v___x_904_; uint8_t v___x_905_; 
v_pos_899_ = lean_ctor_get(v_s_898_, 2);
lean_inc(v_pos_899_);
v___f_900_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__0));
v_s_901_ = l_Lean_Parser_takeWhileFn(v___f_900_, v_c_897_, v_s_898_);
v_pos_902_ = lean_ctor_get(v_s_901_, 2);
lean_inc(v_pos_902_);
v_errorMsg_903_ = lean_ctor_get(v_s_901_, 4);
lean_inc(v_errorMsg_903_);
v___x_904_ = lean_box(0);
v___x_905_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0(v_errorMsg_903_, v___x_904_);
lean_dec(v_errorMsg_903_);
if (v___x_905_ == 0)
{
lean_dec(v_pos_902_);
lean_dec(v_pos_899_);
lean_dec_ref(v_c_897_);
return v_s_901_;
}
else
{
uint8_t v___x_910_; 
v___x_910_ = lean_nat_dec_eq(v_pos_902_, v_pos_899_);
if (v___x_910_ == 0)
{
lean_object* v_toInputContext_911_; lean_object* v_inputString_912_; lean_object* v___x_913_; lean_object* v___x_914_; uint8_t v___x_915_; 
v_toInputContext_911_ = lean_ctor_get(v_c_897_, 0);
v_inputString_912_ = lean_ctor_get(v_toInputContext_911_, 0);
v___x_913_ = lean_string_utf8_extract(v_inputString_912_, v_pos_899_, v_pos_902_);
lean_dec(v_pos_902_);
v___x_914_ = lean_string_data(v___x_913_);
v___x_915_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_lmfdbIdFn_spec__0(v___x_905_, v___x_914_);
lean_dec(v___x_914_);
if (v___x_915_ == 0)
{
lean_dec(v_pos_899_);
lean_dec_ref(v_c_897_);
goto v___jp_906_;
}
else
{
if (v___x_910_ == 0)
{
lean_object* v___x_916_; lean_object* v___x_917_; 
v___x_916_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1));
v___x_917_ = l_Lean_Parser_mkNodeToken(v___x_916_, v_pos_899_, v___x_905_, v_c_897_, v_s_901_);
return v___x_917_;
}
else
{
lean_dec(v_pos_899_);
lean_dec_ref(v_c_897_);
goto v___jp_906_;
}
}
}
else
{
lean_object* v___x_918_; lean_object* v___x_919_; 
lean_dec(v_pos_902_);
lean_dec(v_pos_899_);
lean_dec_ref(v_c_897_);
v___x_918_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__2));
v___x_919_ = l_Lean_Parser_ParserState_mkError(v_s_901_, v___x_918_);
return v___x_919_;
}
}
v___jp_906_:
{
lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_909_; 
v___x_907_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__1));
v___x_908_ = lean_box(0);
v___x_909_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_901_, v___x_907_, v___x_908_, v___x_905_);
return v___x_909_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__0(void){
_start:
{
lean_object* v___x_920_; lean_object* v___x_921_; 
v___x_920_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__0));
v___x_921_ = l_Lean_Parser_mkAtomicInfo(v___x_920_);
return v___x_921_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__1(void){
_start:
{
lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; 
v___x_922_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdFn), 2, 0);
v___x_923_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__0, &lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__0);
v___x_924_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_924_, 0, v___x_923_);
lean_ctor_set(v___x_924_, 1, v___x_922_);
return v___x_924_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot(void){
_start:
{
lean_object* v___x_925_; 
v___x_925_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__1, &lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot___closed__1);
return v___x_925_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__0(void){
_start:
{
uint8_t v___x_926_; uint8_t v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; 
v___x_926_ = 0;
v___x_927_ = 1;
v___x_928_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1));
v___x_929_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__0));
v___x_930_ = l_Lean_Parser_mkAntiquot(v___x_929_, v___x_928_, v___x_927_, v___x_926_);
return v___x_930_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__1(void){
_start:
{
lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; 
v___x_931_ = lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot;
v___x_932_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__0, &lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__0);
v___x_933_ = l_Lean_Parser_withAntiquot(v___x_932_, v___x_931_);
return v___x_933_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdParser(void){
_start:
{
lean_object* v___x_934_; 
v___x_934_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__1, &lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdParser___closed__1);
return v___x_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdFn(lean_object* v_c_943_, lean_object* v_s_944_){
_start:
{
lean_object* v_pos_945_; lean_object* v___f_946_; lean_object* v_s_947_; lean_object* v_pos_948_; lean_object* v_errorMsg_949_; lean_object* v___x_950_; uint8_t v___x_951_; 
v_pos_945_ = lean_ctor_get(v_s_944_, 2);
lean_inc(v_pos_945_);
v___f_946_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagFn___closed__0));
v_s_947_ = l_Lean_Parser_takeWhileFn(v___f_946_, v_c_943_, v_s_944_);
v_pos_948_ = lean_ctor_get(v_s_947_, 2);
lean_inc(v_pos_948_);
v_errorMsg_949_ = lean_ctor_get(v_s_947_, 4);
lean_inc(v_errorMsg_949_);
v___x_950_ = lean_box(0);
v___x_951_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0(v_errorMsg_949_, v___x_950_);
lean_dec(v_errorMsg_949_);
if (v___x_951_ == 0)
{
lean_dec(v_pos_948_);
lean_dec(v_pos_945_);
lean_dec_ref(v_c_943_);
return v_s_947_;
}
else
{
uint8_t v___x_956_; 
v___x_956_ = lean_nat_dec_eq(v_pos_948_, v_pos_945_);
if (v___x_956_ == 0)
{
lean_object* v_toInputContext_957_; lean_object* v_inputString_958_; lean_object* v_id_959_; lean_object* v___x_960_; 
v_toInputContext_957_ = lean_ctor_get(v_c_943_, 0);
v_inputString_958_ = lean_ctor_get(v_toInputContext_957_, 0);
v_id_959_ = lean_string_utf8_extract(v_inputString_958_, v_pos_945_, v_pos_948_);
lean_dec(v_pos_948_);
v___x_960_ = lean_string_data(v_id_959_);
if (lean_obj_tag(v___x_960_) == 1)
{
lean_object* v_head_961_; lean_object* v_tail_962_; uint8_t v___y_981_; uint32_t v___x_985_; uint32_t v___x_986_; uint8_t v___x_987_; 
v_head_961_ = lean_ctor_get(v___x_960_, 0);
lean_inc(v_head_961_);
v_tail_962_ = lean_ctor_get(v___x_960_, 1);
lean_inc(v_tail_962_);
lean_dec_ref_known(v___x_960_, 2);
v___x_985_ = 80;
v___x_986_ = lean_unbox_uint32(v_head_961_);
v___x_987_ = lean_uint32_dec_eq(v___x_986_, v___x_985_);
if (v___x_987_ == 0)
{
v___y_981_ = v___x_951_;
goto v___jp_980_;
}
else
{
v___y_981_ = v___x_956_;
goto v___jp_980_;
}
v___jp_963_:
{
lean_object* v___x_964_; lean_object* v___x_965_; uint8_t v___x_966_; 
v___x_964_ = l_List_lengthTR___redArg(v_tail_962_);
v___x_965_ = lean_unsigned_to_nat(6u);
v___x_966_ = lean_nat_dec_eq(v___x_964_, v___x_965_);
lean_dec(v___x_964_);
if (v___x_966_ == 0)
{
lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; 
lean_dec(v_tail_962_);
lean_dec(v_pos_945_);
lean_dec_ref(v_c_943_);
v___x_967_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__1));
v___x_968_ = lean_box(0);
v___x_969_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_947_, v___x_967_, v___x_968_, v___x_951_);
return v___x_969_;
}
else
{
uint8_t v___x_970_; 
v___x_970_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_wikidataIdFn_spec__0(v_tail_962_);
lean_dec(v_tail_962_);
if (v___x_970_ == 0)
{
lean_object* v___x_971_; lean_object* v___x_972_; lean_object* v___x_973_; 
lean_dec(v_pos_945_);
lean_dec_ref(v_c_943_);
v___x_971_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__2));
v___x_972_ = lean_box(0);
v___x_973_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_947_, v___x_971_, v___x_972_, v___x_951_);
return v___x_973_;
}
else
{
lean_object* v___x_974_; lean_object* v___x_975_; 
v___x_974_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1));
v___x_975_ = l_Lean_Parser_mkNodeToken(v___x_974_, v_pos_945_, v___x_951_, v_c_943_, v_s_947_);
return v___x_975_;
}
}
}
v___jp_976_:
{
uint32_t v___x_977_; uint32_t v___x_978_; uint8_t v___x_979_; 
v___x_977_ = 84;
v___x_978_ = lean_unbox_uint32(v_head_961_);
lean_dec(v_head_961_);
v___x_979_ = lean_uint32_dec_eq(v___x_978_, v___x_977_);
if (v___x_979_ == 0)
{
lean_dec(v_tail_962_);
lean_dec(v_pos_945_);
lean_dec_ref(v_c_943_);
goto v___jp_952_;
}
else
{
if (v___x_956_ == 0)
{
goto v___jp_963_;
}
else
{
lean_dec(v_tail_962_);
lean_dec(v_pos_945_);
lean_dec_ref(v_c_943_);
goto v___jp_952_;
}
}
}
v___jp_980_:
{
if (v___y_981_ == 0)
{
lean_dec(v_head_961_);
goto v___jp_963_;
}
else
{
uint32_t v___x_982_; uint32_t v___x_983_; uint8_t v___x_984_; 
v___x_982_ = 83;
v___x_983_ = lean_unbox_uint32(v_head_961_);
v___x_984_ = lean_uint32_dec_eq(v___x_983_, v___x_982_);
if (v___x_984_ == 0)
{
goto v___jp_976_;
}
else
{
if (v___x_956_ == 0)
{
lean_dec(v_head_961_);
goto v___jp_963_;
}
else
{
goto v___jp_976_;
}
}
}
}
}
else
{
lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; 
lean_dec(v___x_960_);
lean_dec(v_pos_945_);
lean_dec_ref(v_c_943_);
v___x_988_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__2));
v___x_989_ = lean_box(0);
v___x_990_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_947_, v___x_988_, v___x_989_, v___x_951_);
return v___x_990_;
}
}
else
{
lean_object* v___x_991_; lean_object* v___x_992_; 
lean_dec(v_pos_948_);
lean_dec(v_pos_945_);
lean_dec_ref(v_c_943_);
v___x_991_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__3));
v___x_992_ = l_Lean_Parser_ParserState_mkError(v_s_947_, v___x_991_);
return v___x_992_;
}
}
v___jp_952_:
{
lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; 
v___x_953_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdFn___closed__0));
v___x_954_ = lean_box(0);
v___x_955_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_947_, v___x_953_, v___x_954_, v___x_951_);
return v___x_955_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__0(void){
_start:
{
lean_object* v___x_993_; lean_object* v___x_994_; 
v___x_993_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__0));
v___x_994_ = l_Lean_Parser_mkAtomicInfo(v___x_993_);
return v___x_994_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__1(void){
_start:
{
lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; 
v___x_995_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CrossRef_pibaseIdFn), 2, 0);
v___x_996_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__0, &lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__0);
v___x_997_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_997_, 0, v___x_996_);
lean_ctor_set(v___x_997_, 1, v___x_995_);
return v___x_997_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot(void){
_start:
{
lean_object* v___x_998_; 
v___x_998_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__1, &lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot___closed__1);
return v___x_998_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__0(void){
_start:
{
uint8_t v___x_999_; uint8_t v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v___x_999_ = 0;
v___x_1000_ = 1;
v___x_1001_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1));
v___x_1002_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__0));
v___x_1003_ = l_Lean_Parser_mkAntiquot(v___x_1002_, v___x_1001_, v___x_1000_, v___x_999_);
return v___x_1003_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__1(void){
_start:
{
lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; 
v___x_1004_ = lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot;
v___x_1005_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__0, &lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__0);
v___x_1006_ = l_Lean_Parser_withAntiquot(v___x_1005_, v___x_1004_);
return v___x_1006_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_pibaseIdParser(void){
_start:
{
lean_object* v___x_1007_; 
v___x_1007_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__1, &lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_pibaseIdParser___closed__1);
return v___x_1007_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1012_; lean_object* v___f_1013_; 
v___x_1012_ = lean_alloc_closure((void*)(l_instDecidableEqChar___boxed), 2, 0);
v___f_1013_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_1013_, 0, v___x_1012_);
return v___f_1013_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1___boxed__const__1(void){
_start:
{
uint32_t v___x_1014_; lean_object* v___x_1015_; 
v___x_1014_ = 95;
v___x_1015_ = lean_box_uint32(v___x_1014_);
return v___x_1015_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; 
v___x_1016_ = lean_box(0);
v___x_1017_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1___boxed__const__1;
v___x_1018_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1018_, 0, v___x_1017_);
lean_ctor_set(v___x_1018_, 1, v___x_1016_);
return v___x_1018_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2___boxed__const__1(void){
_start:
{
uint32_t v___x_1019_; lean_object* v___x_1020_; 
v___x_1019_ = 46;
v___x_1020_ = lean_box_uint32(v___x_1019_);
return v___x_1020_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2(void){
_start:
{
lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; 
v___x_1021_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1);
v___x_1022_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2___boxed__const__1;
v___x_1023_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1023_, 0, v___x_1022_);
lean_ctor_set(v___x_1023_, 1, v___x_1021_);
return v___x_1023_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3___boxed__const__1(void){
_start:
{
uint32_t v___x_1024_; lean_object* v___x_1025_; 
v___x_1024_ = 84;
v___x_1025_ = lean_box_uint32(v___x_1024_);
return v___x_1025_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3(void){
_start:
{
lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; 
v___x_1026_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2);
v___x_1027_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3___boxed__const__1;
v___x_1028_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1028_, 0, v___x_1027_);
lean_ctor_set(v___x_1028_, 1, v___x_1026_);
return v___x_1028_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4___boxed__const__1(void){
_start:
{
uint32_t v___x_1029_; lean_object* v___x_1030_; 
v___x_1029_ = 70;
v___x_1030_ = lean_box_uint32(v___x_1029_);
return v___x_1030_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4(void){
_start:
{
lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; 
v___x_1031_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3);
v___x_1032_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4___boxed__const__1;
v___x_1033_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1033_, 0, v___x_1032_);
lean_ctor_set(v___x_1033_, 1, v___x_1031_);
return v___x_1033_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5___boxed__const__1(void){
_start:
{
uint32_t v___x_1034_; lean_object* v___x_1035_; 
v___x_1034_ = 69;
v___x_1035_ = lean_box_uint32(v___x_1034_);
return v___x_1035_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5(void){
_start:
{
lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; 
v___x_1036_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4);
v___x_1037_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5___boxed__const__1;
v___x_1038_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1038_, 0, v___x_1037_);
lean_ctor_set(v___x_1038_, 1, v___x_1036_);
return v___x_1038_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6___boxed__const__1(void){
_start:
{
uint32_t v___x_1039_; lean_object* v___x_1040_; 
v___x_1039_ = 109;
v___x_1040_ = lean_box_uint32(v___x_1039_);
return v___x_1040_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6(void){
_start:
{
lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; 
v___x_1041_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5);
v___x_1042_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6___boxed__const__1;
v___x_1043_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1043_, 0, v___x_1042_);
lean_ctor_set(v___x_1043_, 1, v___x_1041_);
return v___x_1043_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7___boxed__const__1(void){
_start:
{
uint32_t v___x_1044_; lean_object* v___x_1045_; 
v___x_1044_ = 100;
v___x_1045_ = lean_box_uint32(v___x_1044_);
return v___x_1045_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7(void){
_start:
{
lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; 
v___x_1046_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6);
v___x_1047_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7___boxed__const__1;
v___x_1048_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1048_, 0, v___x_1047_);
lean_ctor_set(v___x_1048_, 1, v___x_1046_);
return v___x_1048_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8___boxed__const__1(void){
_start:
{
uint32_t v___x_1049_; lean_object* v___x_1050_; 
v___x_1049_ = 99;
v___x_1050_ = lean_box_uint32(v___x_1049_);
return v___x_1050_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8(void){
_start:
{
lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1051_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7);
v___x_1052_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8___boxed__const__1;
v___x_1053_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1053_, 0, v___x_1052_);
lean_ctor_set(v___x_1053_, 1, v___x_1051_);
return v___x_1053_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9___boxed__const__1(void){
_start:
{
uint32_t v___x_1054_; lean_object* v___x_1055_; 
v___x_1054_ = 108;
v___x_1055_ = lean_box_uint32(v___x_1054_);
return v___x_1055_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9(void){
_start:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1056_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8);
v___x_1057_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9___boxed__const__1;
v___x_1058_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1058_, 0, v___x_1057_);
lean_ctor_set(v___x_1058_, 1, v___x_1056_);
return v___x_1058_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10___boxed__const__1(void){
_start:
{
uint32_t v___x_1059_; lean_object* v___x_1060_; 
v___x_1059_ = 120;
v___x_1060_ = lean_box_uint32(v___x_1059_);
return v___x_1060_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10(void){
_start:
{
lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; 
v___x_1061_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9);
v___x_1062_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10___boxed__const__1;
v___x_1063_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1063_, 0, v___x_1062_);
lean_ctor_set(v___x_1063_, 1, v___x_1061_);
return v___x_1063_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11___boxed__const__1(void){
_start:
{
uint32_t v___x_1064_; lean_object* v___x_1065_; 
v___x_1064_ = 118;
v___x_1065_ = lean_box_uint32(v___x_1064_);
return v___x_1065_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11(void){
_start:
{
lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; 
v___x_1066_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10);
v___x_1067_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11___boxed__const__1;
v___x_1068_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1068_, 0, v___x_1067_);
lean_ctor_set(v___x_1068_, 1, v___x_1066_);
return v___x_1068_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12___boxed__const__1(void){
_start:
{
uint32_t v___x_1069_; lean_object* v___x_1070_; 
v___x_1069_ = 105;
v___x_1070_ = lean_box_uint32(v___x_1069_);
return v___x_1070_;
}
}
static lean_object* _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12(void){
_start:
{
lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; 
v___x_1071_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11);
v___x_1072_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12___boxed__const__1;
v___x_1073_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1073_, 0, v___x_1072_);
lean_ctor_set(v___x_1073_, 1, v___x_1071_);
return v___x_1073_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0(uint8_t v___x_1074_, lean_object* v_x_1075_){
_start:
{
if (lean_obj_tag(v_x_1075_) == 0)
{
uint8_t v___x_1076_; 
v___x_1076_ = 1;
return v___x_1076_;
}
else
{
lean_object* v_head_1077_; lean_object* v_tail_1078_; uint8_t v___y_1080_; lean_object* v___f_1082_; lean_object* v___x_1083_; uint8_t v___x_1084_; 
v_head_1077_ = lean_ctor_get(v_x_1075_, 0);
lean_inc_n(v_head_1077_, 2);
v_tail_1078_ = lean_ctor_get(v_x_1075_, 1);
lean_inc(v_tail_1078_);
lean_dec_ref_known(v_x_1075_, 2);
v___f_1082_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__0, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__0_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__0);
v___x_1083_ = lean_obj_once(&lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12, &lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12_once, _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12);
v___x_1084_ = l_List_elem___redArg(v___f_1082_, v_head_1077_, v___x_1083_);
if (v___x_1084_ == 0)
{
uint32_t v___x_1085_; uint32_t v___x_1086_; uint8_t v___x_1087_; 
v___x_1085_ = 48;
v___x_1086_ = lean_unbox_uint32(v_head_1077_);
v___x_1087_ = lean_uint32_dec_le(v___x_1085_, v___x_1086_);
if (v___x_1087_ == 0)
{
lean_dec(v_head_1077_);
v___y_1080_ = v___x_1087_;
goto v___jp_1079_;
}
else
{
uint32_t v___x_1088_; uint32_t v___x_1089_; uint8_t v___x_1090_; 
v___x_1088_ = 57;
v___x_1089_ = lean_unbox_uint32(v_head_1077_);
lean_dec(v_head_1077_);
v___x_1090_ = lean_uint32_dec_le(v___x_1089_, v___x_1088_);
v___y_1080_ = v___x_1090_;
goto v___jp_1079_;
}
}
else
{
lean_dec(v_head_1077_);
v___y_1080_ = v___x_1074_;
goto v___jp_1079_;
}
v___jp_1079_:
{
if (v___y_1080_ == 0)
{
lean_dec(v_tail_1078_);
return v___y_1080_;
}
else
{
v_x_1075_ = v_tail_1078_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___boxed(lean_object* v___x_1091_, lean_object* v_x_1092_){
_start:
{
uint8_t v___x_904__boxed_1093_; uint8_t v_res_1094_; lean_object* v_r_1095_; 
v___x_904__boxed_1093_ = lean_unbox(v___x_1091_);
v_res_1094_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0(v___x_904__boxed_1093_, v_x_1092_);
v_r_1095_ = lean_box(v_res_1094_);
return v_r_1095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdFn(lean_object* v_c_1098_, lean_object* v_s_1099_){
_start:
{
lean_object* v_pos_1100_; lean_object* v___f_1101_; lean_object* v_s_1102_; lean_object* v_pos_1103_; lean_object* v_errorMsg_1104_; lean_object* v___x_1105_; uint8_t v___x_1106_; 
v_pos_1100_ = lean_ctor_get(v_s_1099_, 2);
lean_inc(v_pos_1100_);
v___f_1101_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdFn___closed__0));
v_s_1102_ = l_Lean_Parser_takeWhileFn(v___f_1101_, v_c_1098_, v_s_1099_);
v_pos_1103_ = lean_ctor_get(v_s_1102_, 2);
lean_inc(v_pos_1103_);
v_errorMsg_1104_ = lean_ctor_get(v_s_1102_, 4);
lean_inc(v_errorMsg_1104_);
v___x_1105_ = lean_box(0);
v___x_1106_ = lp_mathlib_Option_instBEq_beq___at___00Mathlib_CrossRef_stacksTagFn_spec__0(v_errorMsg_1104_, v___x_1105_);
lean_dec(v_errorMsg_1104_);
if (v___x_1106_ == 0)
{
lean_dec(v_pos_1103_);
lean_dec(v_pos_1100_);
lean_dec_ref(v_c_1098_);
return v_s_1102_;
}
else
{
uint8_t v___x_1111_; 
v___x_1111_ = lean_nat_dec_eq(v_pos_1103_, v_pos_1100_);
if (v___x_1111_ == 0)
{
lean_object* v_toInputContext_1112_; lean_object* v_inputString_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; uint8_t v___x_1116_; 
v_toInputContext_1112_ = lean_ctor_get(v_c_1098_, 0);
v_inputString_1113_ = lean_ctor_get(v_toInputContext_1112_, 0);
v___x_1114_ = lean_string_utf8_extract(v_inputString_1113_, v_pos_1100_, v_pos_1103_);
lean_dec(v_pos_1103_);
v___x_1115_ = lean_string_data(v___x_1114_);
v___x_1116_ = lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0(v___x_1106_, v___x_1115_);
if (v___x_1116_ == 0)
{
lean_dec(v_pos_1100_);
lean_dec_ref(v_c_1098_);
goto v___jp_1107_;
}
else
{
if (v___x_1111_ == 0)
{
lean_object* v___x_1117_; lean_object* v___x_1118_; 
v___x_1117_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1));
v___x_1118_ = l_Lean_Parser_mkNodeToken(v___x_1117_, v_pos_1100_, v___x_1106_, v_c_1098_, v_s_1102_);
return v___x_1118_;
}
else
{
lean_dec(v_pos_1100_);
lean_dec_ref(v_c_1098_);
goto v___jp_1107_;
}
}
}
else
{
lean_object* v___x_1119_; lean_object* v___x_1120_; 
lean_dec(v_pos_1103_);
lean_dec(v_pos_1100_);
lean_dec_ref(v_c_1098_);
v___x_1119_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdFn___closed__1));
v___x_1120_ = l_Lean_Parser_ParserState_mkError(v_s_1102_, v___x_1119_);
return v___x_1120_;
}
}
v___jp_1107_:
{
lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; 
v___x_1108_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdFn___closed__0));
v___x_1109_ = lean_box(0);
v___x_1110_ = l_Lean_Parser_ParserState_mkUnexpectedError(v_s_1102_, v___x_1108_, v___x_1109_, v___x_1106_);
return v___x_1110_;
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__0(void){
_start:
{
lean_object* v___x_1121_; lean_object* v___x_1122_; 
v___x_1121_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__0));
v___x_1122_ = l_Lean_Parser_mkAtomicInfo(v___x_1121_);
return v___x_1122_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__1(void){
_start:
{
lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; 
v___x_1123_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_CrossRef_dlmfIdFn), 2, 0);
v___x_1124_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__0, &lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__0);
v___x_1125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1125_, 0, v___x_1124_);
lean_ctor_set(v___x_1125_, 1, v___x_1123_);
return v___x_1125_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot(void){
_start:
{
lean_object* v___x_1126_; 
v___x_1126_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__1, &lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot___closed__1);
return v___x_1126_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__0(void){
_start:
{
uint8_t v___x_1127_; uint8_t v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; 
v___x_1127_ = 0;
v___x_1128_ = 1;
v___x_1129_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1));
v___x_1130_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__0));
v___x_1131_ = l_Lean_Parser_mkAntiquot(v___x_1130_, v___x_1129_, v___x_1128_, v___x_1127_);
return v___x_1131_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__1(void){
_start:
{
lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; 
v___x_1132_ = lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot;
v___x_1133_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__0, &lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__0);
v___x_1134_ = l_Lean_Parser_withAntiquot(v___x_1133_, v___x_1132_);
return v___x_1134_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_dlmfIdParser(void){
_start:
{
lean_object* v___x_1135_; 
v___x_1135_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__1, &lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_dlmfIdParser___closed__1);
return v___x_1135_;
}
}
static lean_object* _init_lp_mathlib_Lean_TSyntax_getStacksTag___closed__1(void){
_start:
{
lean_object* v___x_1137_; lean_object* v___x_1138_; 
v___x_1137_ = ((lean_object*)(lp_mathlib_Lean_TSyntax_getStacksTag___closed__0));
v___x_1138_ = l_Lean_stringToMessageData(v___x_1137_);
return v___x_1138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getStacksTag(lean_object* v_stx_1139_, lean_object* v_a_1140_, lean_object* v_a_1141_){
_start:
{
lean_object* v___x_1143_; lean_object* v___x_1144_; 
v___x_1143_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1));
v___x_1144_ = l_Lean_Syntax_isLit_x3f(v___x_1143_, v_stx_1139_);
if (lean_obj_tag(v___x_1144_) == 1)
{
lean_object* v_val_1145_; lean_object* v___x_1147_; uint8_t v_isShared_1148_; uint8_t v_isSharedCheck_1152_; 
v_val_1145_ = lean_ctor_get(v___x_1144_, 0);
v_isSharedCheck_1152_ = !lean_is_exclusive(v___x_1144_);
if (v_isSharedCheck_1152_ == 0)
{
v___x_1147_ = v___x_1144_;
v_isShared_1148_ = v_isSharedCheck_1152_;
goto v_resetjp_1146_;
}
else
{
lean_inc(v_val_1145_);
lean_dec(v___x_1144_);
v___x_1147_ = lean_box(0);
v_isShared_1148_ = v_isSharedCheck_1152_;
goto v_resetjp_1146_;
}
v_resetjp_1146_:
{
lean_object* v___x_1150_; 
if (v_isShared_1148_ == 0)
{
lean_ctor_set_tag(v___x_1147_, 0);
v___x_1150_ = v___x_1147_;
goto v_reusejp_1149_;
}
else
{
lean_object* v_reuseFailAlloc_1151_; 
v_reuseFailAlloc_1151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1151_, 0, v_val_1145_);
v___x_1150_ = v_reuseFailAlloc_1151_;
goto v_reusejp_1149_;
}
v_reusejp_1149_:
{
return v___x_1150_;
}
}
}
else
{
lean_object* v___x_1153_; lean_object* v___x_1154_; 
lean_dec(v___x_1144_);
v___x_1153_ = lean_obj_once(&lp_mathlib_Lean_TSyntax_getStacksTag___closed__1, &lp_mathlib_Lean_TSyntax_getStacksTag___closed__1_once, _init_lp_mathlib_Lean_TSyntax_getStacksTag___closed__1);
v___x_1154_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v___x_1153_, v_a_1140_, v_a_1141_);
return v___x_1154_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getStacksTag___boxed(lean_object* v_stx_1155_, lean_object* v_a_1156_, lean_object* v_a_1157_, lean_object* v_a_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_mathlib_Lean_TSyntax_getStacksTag(v_stx_1155_, v_a_1156_, v_a_1157_);
lean_dec(v_a_1157_);
lean_dec_ref(v_a_1156_);
lean_dec(v_stx_1155_);
return v_res_1159_;
}
}
static lean_object* _init_lp_mathlib_Lean_TSyntax_getWikidataId___closed__1(void){
_start:
{
lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1161_ = ((lean_object*)(lp_mathlib_Lean_TSyntax_getWikidataId___closed__0));
v___x_1162_ = l_Lean_stringToMessageData(v___x_1161_);
return v___x_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getWikidataId(lean_object* v_stx_1163_, lean_object* v_a_1164_, lean_object* v_a_1165_){
_start:
{
lean_object* v___x_1167_; lean_object* v___x_1168_; 
v___x_1167_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1));
v___x_1168_ = l_Lean_Syntax_isLit_x3f(v___x_1167_, v_stx_1163_);
if (lean_obj_tag(v___x_1168_) == 1)
{
lean_object* v_val_1169_; lean_object* v___x_1171_; uint8_t v_isShared_1172_; uint8_t v_isSharedCheck_1176_; 
v_val_1169_ = lean_ctor_get(v___x_1168_, 0);
v_isSharedCheck_1176_ = !lean_is_exclusive(v___x_1168_);
if (v_isSharedCheck_1176_ == 0)
{
v___x_1171_ = v___x_1168_;
v_isShared_1172_ = v_isSharedCheck_1176_;
goto v_resetjp_1170_;
}
else
{
lean_inc(v_val_1169_);
lean_dec(v___x_1168_);
v___x_1171_ = lean_box(0);
v_isShared_1172_ = v_isSharedCheck_1176_;
goto v_resetjp_1170_;
}
v_resetjp_1170_:
{
lean_object* v___x_1174_; 
if (v_isShared_1172_ == 0)
{
lean_ctor_set_tag(v___x_1171_, 0);
v___x_1174_ = v___x_1171_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1175_; 
v_reuseFailAlloc_1175_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1175_, 0, v_val_1169_);
v___x_1174_ = v_reuseFailAlloc_1175_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
return v___x_1174_;
}
}
}
else
{
lean_object* v___x_1177_; lean_object* v___x_1178_; 
lean_dec(v___x_1168_);
v___x_1177_ = lean_obj_once(&lp_mathlib_Lean_TSyntax_getWikidataId___closed__1, &lp_mathlib_Lean_TSyntax_getWikidataId___closed__1_once, _init_lp_mathlib_Lean_TSyntax_getWikidataId___closed__1);
v___x_1178_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v___x_1177_, v_a_1164_, v_a_1165_);
return v___x_1178_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getWikidataId___boxed(lean_object* v_stx_1179_, lean_object* v_a_1180_, lean_object* v_a_1181_, lean_object* v_a_1182_){
_start:
{
lean_object* v_res_1183_; 
v_res_1183_ = lp_mathlib_Lean_TSyntax_getWikidataId(v_stx_1179_, v_a_1180_, v_a_1181_);
lean_dec(v_a_1181_);
lean_dec_ref(v_a_1180_);
lean_dec(v_stx_1179_);
return v_res_1183_;
}
}
static lean_object* _init_lp_mathlib_Lean_TSyntax_getLmfdbId___closed__1(void){
_start:
{
lean_object* v___x_1185_; lean_object* v___x_1186_; 
v___x_1185_ = ((lean_object*)(lp_mathlib_Lean_TSyntax_getLmfdbId___closed__0));
v___x_1186_ = l_Lean_stringToMessageData(v___x_1185_);
return v___x_1186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getLmfdbId(lean_object* v_stx_1187_, lean_object* v_a_1188_, lean_object* v_a_1189_){
_start:
{
lean_object* v___x_1191_; lean_object* v___x_1192_; 
v___x_1191_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1));
v___x_1192_ = l_Lean_Syntax_isLit_x3f(v___x_1191_, v_stx_1187_);
if (lean_obj_tag(v___x_1192_) == 1)
{
lean_object* v_val_1193_; lean_object* v___x_1195_; uint8_t v_isShared_1196_; uint8_t v_isSharedCheck_1200_; 
v_val_1193_ = lean_ctor_get(v___x_1192_, 0);
v_isSharedCheck_1200_ = !lean_is_exclusive(v___x_1192_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_1195_ = v___x_1192_;
v_isShared_1196_ = v_isSharedCheck_1200_;
goto v_resetjp_1194_;
}
else
{
lean_inc(v_val_1193_);
lean_dec(v___x_1192_);
v___x_1195_ = lean_box(0);
v_isShared_1196_ = v_isSharedCheck_1200_;
goto v_resetjp_1194_;
}
v_resetjp_1194_:
{
lean_object* v___x_1198_; 
if (v_isShared_1196_ == 0)
{
lean_ctor_set_tag(v___x_1195_, 0);
v___x_1198_ = v___x_1195_;
goto v_reusejp_1197_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v_val_1193_);
v___x_1198_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1197_;
}
v_reusejp_1197_:
{
return v___x_1198_;
}
}
}
else
{
lean_object* v___x_1201_; lean_object* v___x_1202_; 
lean_dec(v___x_1192_);
v___x_1201_ = lean_obj_once(&lp_mathlib_Lean_TSyntax_getLmfdbId___closed__1, &lp_mathlib_Lean_TSyntax_getLmfdbId___closed__1_once, _init_lp_mathlib_Lean_TSyntax_getLmfdbId___closed__1);
v___x_1202_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v___x_1201_, v_a_1188_, v_a_1189_);
return v___x_1202_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getLmfdbId___boxed(lean_object* v_stx_1203_, lean_object* v_a_1204_, lean_object* v_a_1205_, lean_object* v_a_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_mathlib_Lean_TSyntax_getLmfdbId(v_stx_1203_, v_a_1204_, v_a_1205_);
lean_dec(v_a_1205_);
lean_dec_ref(v_a_1204_);
lean_dec(v_stx_1203_);
return v_res_1207_;
}
}
static lean_object* _init_lp_mathlib_Lean_TSyntax_getPibaseId___closed__1(void){
_start:
{
lean_object* v___x_1209_; lean_object* v___x_1210_; 
v___x_1209_ = ((lean_object*)(lp_mathlib_Lean_TSyntax_getPibaseId___closed__0));
v___x_1210_ = l_Lean_stringToMessageData(v___x_1209_);
return v___x_1210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getPibaseId(lean_object* v_stx_1211_, lean_object* v_a_1212_, lean_object* v_a_1213_){
_start:
{
lean_object* v___x_1215_; lean_object* v___x_1216_; 
v___x_1215_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1));
v___x_1216_ = l_Lean_Syntax_isLit_x3f(v___x_1215_, v_stx_1211_);
if (lean_obj_tag(v___x_1216_) == 1)
{
lean_object* v_val_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1224_; 
v_val_1217_ = lean_ctor_get(v___x_1216_, 0);
v_isSharedCheck_1224_ = !lean_is_exclusive(v___x_1216_);
if (v_isSharedCheck_1224_ == 0)
{
v___x_1219_ = v___x_1216_;
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_val_1217_);
lean_dec(v___x_1216_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1224_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v___x_1222_; 
if (v_isShared_1220_ == 0)
{
lean_ctor_set_tag(v___x_1219_, 0);
v___x_1222_ = v___x_1219_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1223_; 
v_reuseFailAlloc_1223_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1223_, 0, v_val_1217_);
v___x_1222_ = v_reuseFailAlloc_1223_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
return v___x_1222_;
}
}
}
else
{
lean_object* v___x_1225_; lean_object* v___x_1226_; 
lean_dec(v___x_1216_);
v___x_1225_ = lean_obj_once(&lp_mathlib_Lean_TSyntax_getPibaseId___closed__1, &lp_mathlib_Lean_TSyntax_getPibaseId___closed__1_once, _init_lp_mathlib_Lean_TSyntax_getPibaseId___closed__1);
v___x_1226_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v___x_1225_, v_a_1212_, v_a_1213_);
return v___x_1226_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getPibaseId___boxed(lean_object* v_stx_1227_, lean_object* v_a_1228_, lean_object* v_a_1229_, lean_object* v_a_1230_){
_start:
{
lean_object* v_res_1231_; 
v_res_1231_ = lp_mathlib_Lean_TSyntax_getPibaseId(v_stx_1227_, v_a_1228_, v_a_1229_);
lean_dec(v_a_1229_);
lean_dec_ref(v_a_1228_);
lean_dec(v_stx_1227_);
return v_res_1231_;
}
}
static lean_object* _init_lp_mathlib_Lean_TSyntax_getDlmfId___closed__1(void){
_start:
{
lean_object* v___x_1233_; lean_object* v___x_1234_; 
v___x_1233_ = ((lean_object*)(lp_mathlib_Lean_TSyntax_getDlmfId___closed__0));
v___x_1234_ = l_Lean_stringToMessageData(v___x_1233_);
return v___x_1234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getDlmfId(lean_object* v_stx_1235_, lean_object* v_a_1236_, lean_object* v_a_1237_){
_start:
{
lean_object* v___x_1239_; lean_object* v___x_1240_; 
v___x_1239_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1));
v___x_1240_ = l_Lean_Syntax_isLit_x3f(v___x_1239_, v_stx_1235_);
if (lean_obj_tag(v___x_1240_) == 1)
{
lean_object* v_val_1241_; lean_object* v___x_1243_; uint8_t v_isShared_1244_; uint8_t v_isSharedCheck_1248_; 
v_val_1241_ = lean_ctor_get(v___x_1240_, 0);
v_isSharedCheck_1248_ = !lean_is_exclusive(v___x_1240_);
if (v_isSharedCheck_1248_ == 0)
{
v___x_1243_ = v___x_1240_;
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
else
{
lean_inc(v_val_1241_);
lean_dec(v___x_1240_);
v___x_1243_ = lean_box(0);
v_isShared_1244_ = v_isSharedCheck_1248_;
goto v_resetjp_1242_;
}
v_resetjp_1242_:
{
lean_object* v___x_1246_; 
if (v_isShared_1244_ == 0)
{
lean_ctor_set_tag(v___x_1243_, 0);
v___x_1246_ = v___x_1243_;
goto v_reusejp_1245_;
}
else
{
lean_object* v_reuseFailAlloc_1247_; 
v_reuseFailAlloc_1247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1247_, 0, v_val_1241_);
v___x_1246_ = v_reuseFailAlloc_1247_;
goto v_reusejp_1245_;
}
v_reusejp_1245_:
{
return v___x_1246_;
}
}
}
else
{
lean_object* v___x_1249_; lean_object* v___x_1250_; 
lean_dec(v___x_1240_);
v___x_1249_ = lean_obj_once(&lp_mathlib_Lean_TSyntax_getDlmfId___closed__1, &lp_mathlib_Lean_TSyntax_getDlmfId___closed__1_once, _init_lp_mathlib_Lean_TSyntax_getDlmfId___closed__1);
v___x_1250_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v___x_1249_, v_a_1236_, v_a_1237_);
return v___x_1250_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_TSyntax_getDlmfId___boxed(lean_object* v_stx_1251_, lean_object* v_a_1252_, lean_object* v_a_1253_, lean_object* v_a_1254_){
_start:
{
lean_object* v_res_1255_; 
v_res_1255_ = lp_mathlib_Lean_TSyntax_getDlmfId(v_stx_1251_, v_a_1252_, v_a_1253_);
lean_dec(v_a_1253_);
lean_dec_ref(v_a_1252_);
lean_dec(v_stx_1251_);
return v_res_1255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_stacksTagNoAntiquot_formatter(lean_object* v_a_1256_, lean_object* v_a_1257_, lean_object* v_a_1258_, lean_object* v_a_1259_){
_start:
{
lean_object* v___x_1261_; lean_object* v___x_1262_; 
v___x_1261_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagKind___closed__1));
v___x_1262_ = l_Lean_PrettyPrinter_Formatter_visitAtom(v___x_1261_, v_a_1256_, v_a_1257_, v_a_1258_, v_a_1259_);
return v___x_1262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_stacksTagNoAntiquot_formatter___boxed(lean_object* v_a_1263_, lean_object* v_a_1264_, lean_object* v_a_1265_, lean_object* v_a_1266_, lean_object* v_a_1267_){
_start:
{
lean_object* v_res_1268_; 
v_res_1268_ = lp_mathlib_Lean_PrettyPrinter_Formatter_stacksTagNoAntiquot_formatter(v_a_1263_, v_a_1264_, v_a_1265_, v_a_1266_);
lean_dec(v_a_1266_);
lean_dec_ref(v_a_1265_);
lean_dec(v_a_1264_);
lean_dec_ref(v_a_1263_);
return v_res_1268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_wikidataIdNoAntiquot_formatter(lean_object* v_a_1269_, lean_object* v_a_1270_, lean_object* v_a_1271_, lean_object* v_a_1272_){
_start:
{
lean_object* v___x_1274_; lean_object* v___x_1275_; 
v___x_1274_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdKind___closed__1));
v___x_1275_ = l_Lean_PrettyPrinter_Formatter_visitAtom(v___x_1274_, v_a_1269_, v_a_1270_, v_a_1271_, v_a_1272_);
return v___x_1275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_wikidataIdNoAntiquot_formatter___boxed(lean_object* v_a_1276_, lean_object* v_a_1277_, lean_object* v_a_1278_, lean_object* v_a_1279_, lean_object* v_a_1280_){
_start:
{
lean_object* v_res_1281_; 
v_res_1281_ = lp_mathlib_Lean_PrettyPrinter_Formatter_wikidataIdNoAntiquot_formatter(v_a_1276_, v_a_1277_, v_a_1278_, v_a_1279_);
lean_dec(v_a_1279_);
lean_dec_ref(v_a_1278_);
lean_dec(v_a_1277_);
lean_dec_ref(v_a_1276_);
return v_res_1281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_lmfdbIdNoAntiquot_formatter(lean_object* v_a_1282_, lean_object* v_a_1283_, lean_object* v_a_1284_, lean_object* v_a_1285_){
_start:
{
lean_object* v___x_1287_; lean_object* v___x_1288_; 
v___x_1287_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdKind___closed__1));
v___x_1288_ = l_Lean_PrettyPrinter_Formatter_visitAtom(v___x_1287_, v_a_1282_, v_a_1283_, v_a_1284_, v_a_1285_);
return v___x_1288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_lmfdbIdNoAntiquot_formatter___boxed(lean_object* v_a_1289_, lean_object* v_a_1290_, lean_object* v_a_1291_, lean_object* v_a_1292_, lean_object* v_a_1293_){
_start:
{
lean_object* v_res_1294_; 
v_res_1294_ = lp_mathlib_Lean_PrettyPrinter_Formatter_lmfdbIdNoAntiquot_formatter(v_a_1289_, v_a_1290_, v_a_1291_, v_a_1292_);
lean_dec(v_a_1292_);
lean_dec_ref(v_a_1291_);
lean_dec(v_a_1290_);
lean_dec_ref(v_a_1289_);
return v_res_1294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_pibaseIdNoAntiquot_formatter(lean_object* v_a_1295_, lean_object* v_a_1296_, lean_object* v_a_1297_, lean_object* v_a_1298_){
_start:
{
lean_object* v___x_1300_; lean_object* v___x_1301_; 
v___x_1300_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdKind___closed__1));
v___x_1301_ = l_Lean_PrettyPrinter_Formatter_visitAtom(v___x_1300_, v_a_1295_, v_a_1296_, v_a_1297_, v_a_1298_);
return v___x_1301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_pibaseIdNoAntiquot_formatter___boxed(lean_object* v_a_1302_, lean_object* v_a_1303_, lean_object* v_a_1304_, lean_object* v_a_1305_, lean_object* v_a_1306_){
_start:
{
lean_object* v_res_1307_; 
v_res_1307_ = lp_mathlib_Lean_PrettyPrinter_Formatter_pibaseIdNoAntiquot_formatter(v_a_1302_, v_a_1303_, v_a_1304_, v_a_1305_);
lean_dec(v_a_1305_);
lean_dec_ref(v_a_1304_);
lean_dec(v_a_1303_);
lean_dec_ref(v_a_1302_);
return v_res_1307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_dlmfIdNoAntiquot_formatter(lean_object* v_a_1308_, lean_object* v_a_1309_, lean_object* v_a_1310_, lean_object* v_a_1311_){
_start:
{
lean_object* v___x_1313_; lean_object* v___x_1314_; 
v___x_1313_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdKind___closed__1));
v___x_1314_ = l_Lean_PrettyPrinter_Formatter_visitAtom(v___x_1313_, v_a_1308_, v_a_1309_, v_a_1310_, v_a_1311_);
return v___x_1314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Formatter_dlmfIdNoAntiquot_formatter___boxed(lean_object* v_a_1315_, lean_object* v_a_1316_, lean_object* v_a_1317_, lean_object* v_a_1318_, lean_object* v_a_1319_){
_start:
{
lean_object* v_res_1320_; 
v_res_1320_ = lp_mathlib_Lean_PrettyPrinter_Formatter_dlmfIdNoAntiquot_formatter(v_a_1315_, v_a_1316_, v_a_1317_, v_a_1318_);
lean_dec(v_a_1318_);
lean_dec_ref(v_a_1317_);
lean_dec(v_a_1316_);
lean_dec_ref(v_a_1315_);
return v_res_1320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer___redArg(lean_object* v_a_1321_){
_start:
{
lean_object* v___x_1323_; 
v___x_1323_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1321_);
return v___x_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer___redArg___boxed(lean_object* v_a_1324_, lean_object* v_a_1325_){
_start:
{
lean_object* v_res_1326_; 
v_res_1326_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer___redArg(v_a_1324_);
lean_dec(v_a_1324_);
return v_res_1326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer(lean_object* v_a_1327_, lean_object* v_a_1328_, lean_object* v_a_1329_, lean_object* v_a_1330_){
_start:
{
lean_object* v___x_1332_; 
v___x_1332_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1328_);
return v___x_1332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer___boxed(lean_object* v_a_1333_, lean_object* v_a_1334_, lean_object* v_a_1335_, lean_object* v_a_1336_, lean_object* v_a_1337_){
_start:
{
lean_object* v_res_1338_; 
v_res_1338_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_stacksTagAntiquot_parenthesizer(v_a_1333_, v_a_1334_, v_a_1335_, v_a_1336_);
lean_dec(v_a_1336_);
lean_dec_ref(v_a_1335_);
lean_dec(v_a_1334_);
lean_dec_ref(v_a_1333_);
return v_res_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer___redArg(lean_object* v_a_1339_){
_start:
{
lean_object* v___x_1341_; 
v___x_1341_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1339_);
return v___x_1341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer___redArg___boxed(lean_object* v_a_1342_, lean_object* v_a_1343_){
_start:
{
lean_object* v_res_1344_; 
v_res_1344_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer___redArg(v_a_1342_);
lean_dec(v_a_1342_);
return v_res_1344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer(lean_object* v_a_1345_, lean_object* v_a_1346_, lean_object* v_a_1347_, lean_object* v_a_1348_){
_start:
{
lean_object* v___x_1350_; 
v___x_1350_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1346_);
return v___x_1350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer___boxed(lean_object* v_a_1351_, lean_object* v_a_1352_, lean_object* v_a_1353_, lean_object* v_a_1354_, lean_object* v_a_1355_){
_start:
{
lean_object* v_res_1356_; 
v_res_1356_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_wikidataIdAntiquot_parenthesizer(v_a_1351_, v_a_1352_, v_a_1353_, v_a_1354_);
lean_dec(v_a_1354_);
lean_dec_ref(v_a_1353_);
lean_dec(v_a_1352_);
lean_dec_ref(v_a_1351_);
return v_res_1356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer___redArg(lean_object* v_a_1357_){
_start:
{
lean_object* v___x_1359_; 
v___x_1359_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1357_);
return v___x_1359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer___redArg___boxed(lean_object* v_a_1360_, lean_object* v_a_1361_){
_start:
{
lean_object* v_res_1362_; 
v_res_1362_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer___redArg(v_a_1360_);
lean_dec(v_a_1360_);
return v_res_1362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer(lean_object* v_a_1363_, lean_object* v_a_1364_, lean_object* v_a_1365_, lean_object* v_a_1366_){
_start:
{
lean_object* v___x_1368_; 
v___x_1368_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1364_);
return v___x_1368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer___boxed(lean_object* v_a_1369_, lean_object* v_a_1370_, lean_object* v_a_1371_, lean_object* v_a_1372_, lean_object* v_a_1373_){
_start:
{
lean_object* v_res_1374_; 
v_res_1374_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_lmfdbIdAntiquot_parenthesizer(v_a_1369_, v_a_1370_, v_a_1371_, v_a_1372_);
lean_dec(v_a_1372_);
lean_dec_ref(v_a_1371_);
lean_dec(v_a_1370_);
lean_dec_ref(v_a_1369_);
return v_res_1374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer___redArg(lean_object* v_a_1375_){
_start:
{
lean_object* v___x_1377_; 
v___x_1377_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1375_);
return v___x_1377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer___redArg___boxed(lean_object* v_a_1378_, lean_object* v_a_1379_){
_start:
{
lean_object* v_res_1380_; 
v_res_1380_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer___redArg(v_a_1378_);
lean_dec(v_a_1378_);
return v_res_1380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer(lean_object* v_a_1381_, lean_object* v_a_1382_, lean_object* v_a_1383_, lean_object* v_a_1384_){
_start:
{
lean_object* v___x_1386_; 
v___x_1386_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1382_);
return v___x_1386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer___boxed(lean_object* v_a_1387_, lean_object* v_a_1388_, lean_object* v_a_1389_, lean_object* v_a_1390_, lean_object* v_a_1391_){
_start:
{
lean_object* v_res_1392_; 
v_res_1392_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_pibaseIdAntiquot_parenthesizer(v_a_1387_, v_a_1388_, v_a_1389_, v_a_1390_);
lean_dec(v_a_1390_);
lean_dec_ref(v_a_1389_);
lean_dec(v_a_1388_);
lean_dec_ref(v_a_1387_);
return v_res_1392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer___redArg(lean_object* v_a_1393_){
_start:
{
lean_object* v___x_1395_; 
v___x_1395_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1393_);
return v___x_1395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer___redArg___boxed(lean_object* v_a_1396_, lean_object* v_a_1397_){
_start:
{
lean_object* v_res_1398_; 
v_res_1398_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer___redArg(v_a_1396_);
lean_dec(v_a_1396_);
return v_res_1398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer(lean_object* v_a_1399_, lean_object* v_a_1400_, lean_object* v_a_1401_, lean_object* v_a_1402_){
_start:
{
lean_object* v___x_1404_; 
v___x_1404_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v_a_1400_);
return v___x_1404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer___boxed(lean_object* v_a_1405_, lean_object* v_a_1406_, lean_object* v_a_1407_, lean_object* v_a_1408_, lean_object* v_a_1409_){
_start:
{
lean_object* v_res_1410_; 
v_res_1410_ = lp_mathlib_Lean_PrettyPrinter_Parenthesizer_dlmfIdAntiquot_parenthesizer(v_a_1405_, v_a_1406_, v_a_1407_, v_a_1408_);
lean_dec(v_a_1408_);
lean_dec_ref(v_a_1407_);
lean_dec(v_a_1406_);
lean_dec_ref(v_a_1405_);
return v_res_1410_;
}
}
static lean_object* _init_lp_mathlib_Lean_Parser_Category_stacksTagDB(void){
_start:
{
lean_object* v___x_1454_; 
v___x_1454_ = lean_box(0);
return v___x_1454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter(lean_object* v_a_1530_, lean_object* v_a_1531_, lean_object* v_a_1532_, lean_object* v_a_1533_){
_start:
{
lean_object* v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; 
v___x_1535_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter___closed__0));
v___x_1536_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Formatter_stacksTagNoAntiquot_formatter___boxed), 5, 0);
v___x_1537_ = l_Lean_PrettyPrinter_Formatter_orelse_formatter(v___x_1535_, v___x_1536_, v_a_1530_, v_a_1531_, v_a_1532_, v_a_1533_);
return v___x_1537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter___boxed(lean_object* v_a_1538_, lean_object* v_a_1539_, lean_object* v_a_1540_, lean_object* v_a_1541_, lean_object* v_a_1542_){
_start:
{
lean_object* v_res_1543_; 
v_res_1543_ = lp_mathlib_Mathlib_CrossRef_stacksTagParser_formatter(v_a_1538_, v_a_1539_, v_a_1540_, v_a_1541_);
lean_dec(v_a_1541_);
lean_dec_ref(v_a_1540_);
lean_dec(v_a_1539_);
lean_dec_ref(v_a_1538_);
return v_res_1543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___lam__0(lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_){
_start:
{
lean_object* v___x_1549_; 
v___x_1549_ = l_Lean_PrettyPrinter_Parenthesizer_visitToken___redArg(v___y_1545_);
return v___x_1549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___lam__0___boxed(lean_object* v___y_1550_, lean_object* v___y_1551_, lean_object* v___y_1552_, lean_object* v___y_1553_, lean_object* v___y_1554_){
_start:
{
lean_object* v_res_1555_; 
v_res_1555_ = lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___lam__0(v___y_1550_, v___y_1551_, v___y_1552_, v___y_1553_);
lean_dec(v___y_1553_);
lean_dec_ref(v___y_1552_);
lean_dec(v___y_1551_);
lean_dec_ref(v___y_1550_);
return v_res_1555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer(lean_object* v_a_1564_, lean_object* v_a_1565_, lean_object* v_a_1566_, lean_object* v_a_1567_){
_start:
{
lean_object* v___f_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; 
v___f_1569_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__0));
v___x_1570_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__1));
v___x_1571_ = l_Lean_PrettyPrinter_Parenthesizer_withAntiquot_parenthesizer(v___x_1570_, v___f_1569_, v_a_1564_, v_a_1565_, v_a_1566_, v_a_1567_);
return v___x_1571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___boxed(lean_object* v_a_1572_, lean_object* v_a_1573_, lean_object* v_a_1574_, lean_object* v_a_1575_, lean_object* v_a_1576_){
_start:
{
lean_object* v_res_1577_; 
v_res_1577_ = lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer(v_a_1572_, v_a_1573_, v_a_1574_, v_a_1575_);
lean_dec(v_a_1575_);
lean_dec_ref(v_a_1574_);
lean_dec(v_a_1573_);
lean_dec_ref(v_a_1572_);
return v_res_1577_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; 
v___x_1578_ = lean_box(0);
v___x_1579_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1580_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1580_, 0, v___x_1579_);
lean_ctor_set(v___x_1580_, 1, v___x_1578_);
return v___x_1580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg(){
_start:
{
lean_object* v___x_1582_; lean_object* v___x_1583_; 
v___x_1582_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_1583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1583_, 0, v___x_1582_);
return v___x_1583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___boxed(lean_object* v___y_1584_){
_start:
{
lean_object* v_res_1585_; 
v_res_1585_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
return v_res_1585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0(lean_object* v_00_u03b1_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_){
_start:
{
lean_object* v___x_1590_; 
v___x_1590_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
return v___x_1590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___boxed(lean_object* v_00_u03b1_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_){
_start:
{
lean_object* v_res_1595_; 
v_res_1595_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0(v_00_u03b1_1591_, v___y_1592_, v___y_1593_);
lean_dec(v___y_1593_);
lean_dec_ref(v___y_1592_);
return v_res_1595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(lean_object* v___x_1596_, lean_object* v___x_1597_, lean_object* v___x_1598_, lean_object* v___x_1599_, lean_object* v___x_1600_, lean_object* v_decl_1601_, lean_object* v_stx_1602_, uint8_t v___attrKind_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_){
_start:
{
lean_object* v_fst_1608_; lean_object* v_fst_1609_; lean_object* v_snd_1610_; lean_object* v___y_1611_; lean_object* v___y_1612_; lean_object* v___x_1629_; uint8_t v___x_1630_; 
lean_inc_ref(v___x_1597_);
lean_inc_ref(v___x_1596_);
v___x_1629_ = l_Lean_Name_mkStr3(v___x_1596_, v___x_1597_, v___x_1598_);
lean_inc(v_stx_1602_);
v___x_1630_ = l_Lean_Syntax_isOfKind(v_stx_1602_, v___x_1629_);
lean_dec(v___x_1629_);
if (v___x_1630_ == 0)
{
lean_object* v___x_1631_; lean_object* v_a_1632_; lean_object* v___x_1634_; uint8_t v_isShared_1635_; uint8_t v_isSharedCheck_1639_; 
lean_dec(v_stx_1602_);
lean_dec(v_decl_1601_);
lean_dec_ref(v___x_1597_);
lean_dec_ref(v___x_1596_);
v___x_1631_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_1632_ = lean_ctor_get(v___x_1631_, 0);
v_isSharedCheck_1639_ = !lean_is_exclusive(v___x_1631_);
if (v_isSharedCheck_1639_ == 0)
{
v___x_1634_ = v___x_1631_;
v_isShared_1635_ = v_isSharedCheck_1639_;
goto v_resetjp_1633_;
}
else
{
lean_inc(v_a_1632_);
lean_dec(v___x_1631_);
v___x_1634_ = lean_box(0);
v_isShared_1635_ = v_isSharedCheck_1639_;
goto v_resetjp_1633_;
}
v_resetjp_1633_:
{
lean_object* v___x_1637_; 
if (v_isShared_1635_ == 0)
{
v___x_1637_ = v___x_1634_;
goto v_reusejp_1636_;
}
else
{
lean_object* v_reuseFailAlloc_1638_; 
v_reuseFailAlloc_1638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1638_, 0, v_a_1632_);
v___x_1637_ = v_reuseFailAlloc_1638_;
goto v_reusejp_1636_;
}
v_reusejp_1636_:
{
return v___x_1637_;
}
}
}
else
{
lean_object* v___x_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; uint8_t v___x_1643_; 
v___x_1640_ = l_Lean_Syntax_getArg(v_stx_1602_, v___x_1599_);
v___x_1641_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagDBStacks___closed__0));
lean_inc_ref(v___x_1597_);
lean_inc_ref(v___x_1596_);
v___x_1642_ = l_Lean_Name_mkStr3(v___x_1596_, v___x_1597_, v___x_1641_);
lean_inc(v___x_1640_);
v___x_1643_ = l_Lean_Syntax_isOfKind(v___x_1640_, v___x_1642_);
lean_dec(v___x_1642_);
if (v___x_1643_ == 0)
{
lean_object* v___x_1644_; lean_object* v___x_1645_; uint8_t v___x_1646_; 
v___x_1644_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagDBKerodon___closed__0));
v___x_1645_ = l_Lean_Name_mkStr3(v___x_1596_, v___x_1597_, v___x_1644_);
v___x_1646_ = l_Lean_Syntax_isOfKind(v___x_1640_, v___x_1645_);
lean_dec(v___x_1645_);
if (v___x_1646_ == 0)
{
lean_object* v___x_1647_; lean_object* v_a_1648_; lean_object* v___x_1650_; uint8_t v_isShared_1651_; uint8_t v_isSharedCheck_1655_; 
lean_dec(v_stx_1602_);
lean_dec(v_decl_1601_);
v___x_1647_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_1648_ = lean_ctor_get(v___x_1647_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1647_);
if (v_isSharedCheck_1655_ == 0)
{
v___x_1650_ = v___x_1647_;
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
else
{
lean_inc(v_a_1648_);
lean_dec(v___x_1647_);
v___x_1650_ = lean_box(0);
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
v_resetjp_1649_:
{
lean_object* v___x_1653_; 
if (v_isShared_1651_ == 0)
{
v___x_1653_ = v___x_1650_;
goto v_reusejp_1652_;
}
else
{
lean_object* v_reuseFailAlloc_1654_; 
v_reuseFailAlloc_1654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1654_, 0, v_a_1648_);
v___x_1653_ = v_reuseFailAlloc_1654_;
goto v_reusejp_1652_;
}
v_reusejp_1652_:
{
return v___x_1653_;
}
}
}
else
{
lean_object* v___x_1656_; lean_object* v_tag_1657_; lean_object* v_comment_1659_; lean_object* v___y_1660_; lean_object* v___y_1661_; lean_object* v___x_1663_; uint8_t v___x_1664_; 
v___x_1656_ = lean_unsigned_to_nat(1u);
v_tag_1657_ = l_Lean_Syntax_getArg(v_stx_1602_, v___x_1656_);
v___x_1663_ = l_Lean_Syntax_getArg(v_stx_1602_, v___x_1600_);
lean_dec(v_stx_1602_);
v___x_1664_ = l_Lean_Syntax_isNone(v___x_1663_);
if (v___x_1664_ == 0)
{
uint8_t v___x_1665_; 
lean_inc(v___x_1663_);
v___x_1665_ = l_Lean_Syntax_matchesNull(v___x_1663_, v___x_1656_);
if (v___x_1665_ == 0)
{
lean_object* v___x_1666_; lean_object* v_a_1667_; lean_object* v___x_1669_; uint8_t v_isShared_1670_; uint8_t v_isSharedCheck_1674_; 
lean_dec(v___x_1663_);
lean_dec(v_tag_1657_);
lean_dec(v_decl_1601_);
v___x_1666_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_1667_ = lean_ctor_get(v___x_1666_, 0);
v_isSharedCheck_1674_ = !lean_is_exclusive(v___x_1666_);
if (v_isSharedCheck_1674_ == 0)
{
v___x_1669_ = v___x_1666_;
v_isShared_1670_ = v_isSharedCheck_1674_;
goto v_resetjp_1668_;
}
else
{
lean_inc(v_a_1667_);
lean_dec(v___x_1666_);
v___x_1669_ = lean_box(0);
v_isShared_1670_ = v_isSharedCheck_1674_;
goto v_resetjp_1668_;
}
v_resetjp_1668_:
{
lean_object* v___x_1672_; 
if (v_isShared_1670_ == 0)
{
v___x_1672_ = v___x_1669_;
goto v_reusejp_1671_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v_a_1667_);
v___x_1672_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1671_;
}
v_reusejp_1671_:
{
return v___x_1672_;
}
}
}
else
{
lean_object* v_comment_1675_; lean_object* v___x_1676_; 
v_comment_1675_ = l_Lean_Syntax_getArg(v___x_1663_, v___x_1599_);
lean_dec(v___x_1663_);
v___x_1676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1676_, 0, v_comment_1675_);
v_comment_1659_ = v___x_1676_;
v___y_1660_ = v___y_1604_;
v___y_1661_ = v___y_1605_;
goto v___jp_1658_;
}
}
else
{
lean_object* v___x_1677_; 
lean_dec(v___x_1663_);
v___x_1677_ = lean_box(0);
v_comment_1659_ = v___x_1677_;
v___y_1660_ = v___y_1604_;
v___y_1661_ = v___y_1605_;
goto v___jp_1658_;
}
v___jp_1658_:
{
lean_object* v___x_1662_; 
v___x_1662_ = lean_box(1);
v_fst_1608_ = v___x_1662_;
v_fst_1609_ = v_tag_1657_;
v_snd_1610_ = v_comment_1659_;
v___y_1611_ = v___y_1660_;
v___y_1612_ = v___y_1661_;
goto v___jp_1607_;
}
}
}
else
{
lean_object* v___x_1678_; lean_object* v_tag_1679_; lean_object* v_comment_1681_; lean_object* v___y_1682_; lean_object* v___y_1683_; lean_object* v___x_1685_; uint8_t v___x_1686_; 
lean_dec(v___x_1640_);
lean_dec_ref(v___x_1597_);
lean_dec_ref(v___x_1596_);
v___x_1678_ = lean_unsigned_to_nat(1u);
v_tag_1679_ = l_Lean_Syntax_getArg(v_stx_1602_, v___x_1678_);
v___x_1685_ = l_Lean_Syntax_getArg(v_stx_1602_, v___x_1600_);
lean_dec(v_stx_1602_);
v___x_1686_ = l_Lean_Syntax_isNone(v___x_1685_);
if (v___x_1686_ == 0)
{
uint8_t v___x_1687_; 
lean_inc(v___x_1685_);
v___x_1687_ = l_Lean_Syntax_matchesNull(v___x_1685_, v___x_1678_);
if (v___x_1687_ == 0)
{
lean_object* v___x_1688_; lean_object* v_a_1689_; lean_object* v___x_1691_; uint8_t v_isShared_1692_; uint8_t v_isSharedCheck_1696_; 
lean_dec(v___x_1685_);
lean_dec(v_tag_1679_);
lean_dec(v_decl_1601_);
v___x_1688_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_1689_ = lean_ctor_get(v___x_1688_, 0);
v_isSharedCheck_1696_ = !lean_is_exclusive(v___x_1688_);
if (v_isSharedCheck_1696_ == 0)
{
v___x_1691_ = v___x_1688_;
v_isShared_1692_ = v_isSharedCheck_1696_;
goto v_resetjp_1690_;
}
else
{
lean_inc(v_a_1689_);
lean_dec(v___x_1688_);
v___x_1691_ = lean_box(0);
v_isShared_1692_ = v_isSharedCheck_1696_;
goto v_resetjp_1690_;
}
v_resetjp_1690_:
{
lean_object* v___x_1694_; 
if (v_isShared_1692_ == 0)
{
v___x_1694_ = v___x_1691_;
goto v_reusejp_1693_;
}
else
{
lean_object* v_reuseFailAlloc_1695_; 
v_reuseFailAlloc_1695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1695_, 0, v_a_1689_);
v___x_1694_ = v_reuseFailAlloc_1695_;
goto v_reusejp_1693_;
}
v_reusejp_1693_:
{
return v___x_1694_;
}
}
}
else
{
lean_object* v_comment_1697_; lean_object* v___x_1698_; 
v_comment_1697_ = l_Lean_Syntax_getArg(v___x_1685_, v___x_1599_);
lean_dec(v___x_1685_);
v___x_1698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1698_, 0, v_comment_1697_);
v_comment_1681_ = v___x_1698_;
v___y_1682_ = v___y_1604_;
v___y_1683_ = v___y_1605_;
goto v___jp_1680_;
}
}
else
{
lean_object* v___x_1699_; 
lean_dec(v___x_1685_);
v___x_1699_ = lean_box(0);
v_comment_1681_ = v___x_1699_;
v___y_1682_ = v___y_1604_;
v___y_1683_ = v___y_1605_;
goto v___jp_1680_;
}
v___jp_1680_:
{
lean_object* v___x_1684_; 
v___x_1684_ = lean_box(4);
v_fst_1608_ = v___x_1684_;
v_fst_1609_ = v_tag_1679_;
v_snd_1610_ = v_comment_1681_;
v___y_1611_ = v___y_1682_;
v___y_1612_ = v___y_1683_;
goto v___jp_1607_;
}
}
}
v___jp_1607_:
{
lean_object* v___x_1613_; 
v___x_1613_ = lp_mathlib_Lean_TSyntax_getStacksTag(v_fst_1609_, v___y_1611_, v___y_1612_);
lean_dec(v_fst_1609_);
if (lean_obj_tag(v___x_1613_) == 0)
{
if (lean_obj_tag(v_snd_1610_) == 0)
{
lean_object* v_a_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; 
v_a_1614_ = lean_ctor_get(v___x_1613_, 0);
lean_inc(v_a_1614_);
lean_dec_ref_known(v___x_1613_, 1);
v___x_1615_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___x_1616_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v_fst_1608_, v_decl_1601_, v_a_1614_, v___x_1615_, v___y_1611_, v___y_1612_);
return v___x_1616_;
}
else
{
lean_object* v_a_1617_; lean_object* v_val_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; 
v_a_1617_ = lean_ctor_get(v___x_1613_, 0);
lean_inc(v_a_1617_);
lean_dec_ref_known(v___x_1613_, 1);
v_val_1618_ = lean_ctor_get(v_snd_1610_, 0);
lean_inc(v_val_1618_);
lean_dec_ref_known(v_snd_1610_, 1);
v___x_1619_ = l_Lean_TSyntax_getString(v_val_1618_);
lean_dec(v_val_1618_);
v___x_1620_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v_fst_1608_, v_decl_1601_, v_a_1617_, v___x_1619_, v___y_1611_, v___y_1612_);
return v___x_1620_;
}
}
else
{
lean_object* v_a_1621_; lean_object* v___x_1623_; uint8_t v_isShared_1624_; uint8_t v_isSharedCheck_1628_; 
lean_dec(v_snd_1610_);
lean_dec(v_fst_1608_);
lean_dec(v_decl_1601_);
v_a_1621_ = lean_ctor_get(v___x_1613_, 0);
v_isSharedCheck_1628_ = !lean_is_exclusive(v___x_1613_);
if (v_isSharedCheck_1628_ == 0)
{
v___x_1623_ = v___x_1613_;
v_isShared_1624_ = v_isSharedCheck_1628_;
goto v_resetjp_1622_;
}
else
{
lean_inc(v_a_1621_);
lean_dec(v___x_1613_);
v___x_1623_ = lean_box(0);
v_isShared_1624_ = v_isSharedCheck_1628_;
goto v_resetjp_1622_;
}
v_resetjp_1622_:
{
lean_object* v___x_1626_; 
if (v_isShared_1624_ == 0)
{
v___x_1626_ = v___x_1623_;
goto v_reusejp_1625_;
}
else
{
lean_object* v_reuseFailAlloc_1627_; 
v_reuseFailAlloc_1627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1627_, 0, v_a_1621_);
v___x_1626_ = v_reuseFailAlloc_1627_;
goto v_reusejp_1625_;
}
v_reusejp_1625_:
{
return v___x_1626_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed(lean_object* v___x_1700_, lean_object* v___x_1701_, lean_object* v___x_1702_, lean_object* v___x_1703_, lean_object* v___x_1704_, lean_object* v_decl_1705_, lean_object* v_stx_1706_, lean_object* v___attrKind_1707_, lean_object* v___y_1708_, lean_object* v___y_1709_, lean_object* v___y_1710_){
_start:
{
uint8_t v___attrKind_boxed_1711_; lean_object* v_res_1712_; 
v___attrKind_boxed_1711_ = lean_unbox(v___attrKind_1707_);
v_res_1712_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(v___x_1700_, v___x_1701_, v___x_1702_, v___x_1703_, v___x_1704_, v_decl_1705_, v_stx_1706_, v___attrKind_boxed_1711_, v___y_1708_, v___y_1709_);
lean_dec(v___y_1709_);
lean_dec_ref(v___y_1708_);
lean_dec(v___x_1704_);
lean_dec(v___x_1703_);
return v_res_1712_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1714_; lean_object* v___x_1715_; 
v___x_1714_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_1715_ = l_Lean_stringToMessageData(v___x_1714_);
return v___x_1715_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1717_; lean_object* v___x_1718_; 
v___x_1717_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_1718_ = l_Lean_stringToMessageData(v___x_1717_);
return v___x_1718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(lean_object* v___x_1719_, lean_object* v_decl_1720_, lean_object* v___y_1721_, lean_object* v___y_1722_){
_start:
{
lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; 
v___x_1724_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_);
v___x_1725_ = l_Lean_MessageData_ofName(v___x_1719_);
v___x_1726_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1726_, 0, v___x_1724_);
lean_ctor_set(v___x_1726_, 1, v___x_1725_);
v___x_1727_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_);
v___x_1728_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1728_, 0, v___x_1726_);
lean_ctor_set(v___x_1728_, 1, v___x_1727_);
v___x_1729_ = lp_mathlib_Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1___redArg(v___x_1728_, v___y_1721_, v___y_1722_);
return v___x_1729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed(lean_object* v___x_1730_, lean_object* v_decl_1731_, lean_object* v___y_1732_, lean_object* v___y_1733_, lean_object* v___y_1734_){
_start:
{
lean_object* v_res_1735_; 
v_res_1735_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(v___x_1730_, v_decl_1731_, v___y_1732_, v___y_1733_);
lean_dec(v___y_1733_);
lean_dec_ref(v___y_1732_);
lean_dec(v_decl_1731_);
return v_res_1735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1810_; lean_object* v___x_1811_; 
v___x_1810_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__27_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_1811_ = l_Lean_registerBuiltinAttribute(v___x_1810_);
return v___x_1811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2____boxed(lean_object* v_a_1812_){
_start:
{
lean_object* v_res_1813_; 
v_res_1813_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_();
return v_res_1813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter(lean_object* v_a_1849_, lean_object* v_a_1850_, lean_object* v_a_1851_, lean_object* v_a_1852_){
_start:
{
lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; 
v___x_1854_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter___closed__0));
v___x_1855_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Formatter_wikidataIdNoAntiquot_formatter___boxed), 5, 0);
v___x_1856_ = l_Lean_PrettyPrinter_Formatter_orelse_formatter(v___x_1854_, v___x_1855_, v_a_1849_, v_a_1850_, v_a_1851_, v_a_1852_);
return v___x_1856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter___boxed(lean_object* v_a_1857_, lean_object* v_a_1858_, lean_object* v_a_1859_, lean_object* v_a_1860_, lean_object* v_a_1861_){
_start:
{
lean_object* v_res_1862_; 
v_res_1862_ = lp_mathlib_Mathlib_CrossRef_wikidataIdParser_formatter(v_a_1857_, v_a_1858_, v_a_1859_, v_a_1860_);
lean_dec(v_a_1860_);
lean_dec_ref(v_a_1859_);
lean_dec(v_a_1858_);
lean_dec_ref(v_a_1857_);
return v_res_1862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer(lean_object* v_a_1870_, lean_object* v_a_1871_, lean_object* v_a_1872_, lean_object* v_a_1873_){
_start:
{
lean_object* v___f_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; 
v___f_1875_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__0));
v___x_1876_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer___closed__0));
v___x_1877_ = l_Lean_PrettyPrinter_Parenthesizer_withAntiquot_parenthesizer(v___x_1876_, v___f_1875_, v_a_1870_, v_a_1871_, v_a_1872_, v_a_1873_);
return v___x_1877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer___boxed(lean_object* v_a_1878_, lean_object* v_a_1879_, lean_object* v_a_1880_, lean_object* v_a_1881_, lean_object* v_a_1882_){
_start:
{
lean_object* v_res_1883_; 
v_res_1883_ = lp_mathlib_Mathlib_CrossRef_wikidataIdParser_parenthesizer(v_a_1878_, v_a_1879_, v_a_1880_, v_a_1881_);
lean_dec(v_a_1881_);
lean_dec_ref(v_a_1880_);
lean_dec(v_a_1879_);
lean_dec_ref(v_a_1878_);
return v_res_1883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(lean_object* v___x_1884_, lean_object* v___x_1885_, lean_object* v___x_1886_, lean_object* v___x_1887_, lean_object* v___x_1888_, lean_object* v_decl_1889_, lean_object* v_stx_1890_, uint8_t v___attrKind_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_){
_start:
{
lean_object* v_fst_1896_; lean_object* v_snd_1897_; lean_object* v___y_1898_; lean_object* v___y_1899_; lean_object* v___x_1916_; uint8_t v___x_1917_; 
v___x_1916_ = l_Lean_Name_mkStr3(v___x_1884_, v___x_1885_, v___x_1886_);
lean_inc(v_stx_1890_);
v___x_1917_ = l_Lean_Syntax_isOfKind(v_stx_1890_, v___x_1916_);
lean_dec(v___x_1916_);
if (v___x_1917_ == 0)
{
lean_object* v___x_1918_; lean_object* v_a_1919_; lean_object* v___x_1921_; uint8_t v_isShared_1922_; uint8_t v_isSharedCheck_1926_; 
lean_dec(v_stx_1890_);
lean_dec(v_decl_1889_);
v___x_1918_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_1919_ = lean_ctor_get(v___x_1918_, 0);
v_isSharedCheck_1926_ = !lean_is_exclusive(v___x_1918_);
if (v_isSharedCheck_1926_ == 0)
{
v___x_1921_ = v___x_1918_;
v_isShared_1922_ = v_isSharedCheck_1926_;
goto v_resetjp_1920_;
}
else
{
lean_inc(v_a_1919_);
lean_dec(v___x_1918_);
v___x_1921_ = lean_box(0);
v_isShared_1922_ = v_isSharedCheck_1926_;
goto v_resetjp_1920_;
}
v_resetjp_1920_:
{
lean_object* v___x_1924_; 
if (v_isShared_1922_ == 0)
{
v___x_1924_ = v___x_1921_;
goto v_reusejp_1923_;
}
else
{
lean_object* v_reuseFailAlloc_1925_; 
v_reuseFailAlloc_1925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1925_, 0, v_a_1919_);
v___x_1924_ = v_reuseFailAlloc_1925_;
goto v_reusejp_1923_;
}
v_reusejp_1923_:
{
return v___x_1924_;
}
}
}
else
{
lean_object* v___x_1927_; lean_object* v_id_1928_; lean_object* v___x_1929_; uint8_t v___x_1930_; 
v___x_1927_ = lean_unsigned_to_nat(1u);
v_id_1928_ = l_Lean_Syntax_getArg(v_stx_1890_, v___x_1927_);
v___x_1929_ = l_Lean_Syntax_getArg(v_stx_1890_, v___x_1887_);
lean_dec(v_stx_1890_);
v___x_1930_ = l_Lean_Syntax_isNone(v___x_1929_);
if (v___x_1930_ == 0)
{
uint8_t v___x_1931_; 
lean_inc(v___x_1929_);
v___x_1931_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1927_);
if (v___x_1931_ == 0)
{
lean_object* v___x_1932_; lean_object* v_a_1933_; lean_object* v___x_1935_; uint8_t v_isShared_1936_; uint8_t v_isSharedCheck_1940_; 
lean_dec(v___x_1929_);
lean_dec(v_id_1928_);
lean_dec(v_decl_1889_);
v___x_1932_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_1933_ = lean_ctor_get(v___x_1932_, 0);
v_isSharedCheck_1940_ = !lean_is_exclusive(v___x_1932_);
if (v_isSharedCheck_1940_ == 0)
{
v___x_1935_ = v___x_1932_;
v_isShared_1936_ = v_isSharedCheck_1940_;
goto v_resetjp_1934_;
}
else
{
lean_inc(v_a_1933_);
lean_dec(v___x_1932_);
v___x_1935_ = lean_box(0);
v_isShared_1936_ = v_isSharedCheck_1940_;
goto v_resetjp_1934_;
}
v_resetjp_1934_:
{
lean_object* v___x_1938_; 
if (v_isShared_1936_ == 0)
{
v___x_1938_ = v___x_1935_;
goto v_reusejp_1937_;
}
else
{
lean_object* v_reuseFailAlloc_1939_; 
v_reuseFailAlloc_1939_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1939_, 0, v_a_1933_);
v___x_1938_ = v_reuseFailAlloc_1939_;
goto v_reusejp_1937_;
}
v_reusejp_1937_:
{
return v___x_1938_;
}
}
}
else
{
lean_object* v_comment_1941_; lean_object* v___x_1942_; 
v_comment_1941_ = l_Lean_Syntax_getArg(v___x_1929_, v___x_1888_);
lean_dec(v___x_1929_);
v___x_1942_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1942_, 0, v_comment_1941_);
v_fst_1896_ = v_id_1928_;
v_snd_1897_ = v___x_1942_;
v___y_1898_ = v___y_1892_;
v___y_1899_ = v___y_1893_;
goto v___jp_1895_;
}
}
else
{
lean_object* v___x_1943_; 
lean_dec(v___x_1929_);
v___x_1943_ = lean_box(0);
v_fst_1896_ = v_id_1928_;
v_snd_1897_ = v___x_1943_;
v___y_1898_ = v___y_1892_;
v___y_1899_ = v___y_1893_;
goto v___jp_1895_;
}
}
v___jp_1895_:
{
lean_object* v___x_1900_; 
v___x_1900_ = lp_mathlib_Lean_TSyntax_getWikidataId(v_fst_1896_, v___y_1898_, v___y_1899_);
lean_dec(v_fst_1896_);
if (lean_obj_tag(v___x_1900_) == 0)
{
lean_object* v_a_1901_; lean_object* v___x_1902_; 
v_a_1901_ = lean_ctor_get(v___x_1900_, 0);
lean_inc(v_a_1901_);
lean_dec_ref_known(v___x_1900_, 1);
v___x_1902_ = lean_box(5);
if (lean_obj_tag(v_snd_1897_) == 0)
{
lean_object* v___x_1903_; lean_object* v___x_1904_; 
v___x_1903_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___x_1904_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v___x_1902_, v_decl_1889_, v_a_1901_, v___x_1903_, v___y_1898_, v___y_1899_);
return v___x_1904_;
}
else
{
lean_object* v_val_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; 
v_val_1905_ = lean_ctor_get(v_snd_1897_, 0);
lean_inc(v_val_1905_);
lean_dec_ref_known(v_snd_1897_, 1);
v___x_1906_ = l_Lean_TSyntax_getString(v_val_1905_);
lean_dec(v_val_1905_);
v___x_1907_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v___x_1902_, v_decl_1889_, v_a_1901_, v___x_1906_, v___y_1898_, v___y_1899_);
return v___x_1907_;
}
}
else
{
lean_object* v_a_1908_; lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_1915_; 
lean_dec(v_snd_1897_);
lean_dec(v_decl_1889_);
v_a_1908_ = lean_ctor_get(v___x_1900_, 0);
v_isSharedCheck_1915_ = !lean_is_exclusive(v___x_1900_);
if (v_isSharedCheck_1915_ == 0)
{
v___x_1910_ = v___x_1900_;
v_isShared_1911_ = v_isSharedCheck_1915_;
goto v_resetjp_1909_;
}
else
{
lean_inc(v_a_1908_);
lean_dec(v___x_1900_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_1915_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
lean_object* v___x_1913_; 
if (v_isShared_1911_ == 0)
{
v___x_1913_ = v___x_1910_;
goto v_reusejp_1912_;
}
else
{
lean_object* v_reuseFailAlloc_1914_; 
v_reuseFailAlloc_1914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1914_, 0, v_a_1908_);
v___x_1913_ = v_reuseFailAlloc_1914_;
goto v_reusejp_1912_;
}
v_reusejp_1912_:
{
return v___x_1913_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2____boxed(lean_object* v___x_1944_, lean_object* v___x_1945_, lean_object* v___x_1946_, lean_object* v___x_1947_, lean_object* v___x_1948_, lean_object* v_decl_1949_, lean_object* v_stx_1950_, lean_object* v___attrKind_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_){
_start:
{
uint8_t v___attrKind_boxed_1955_; lean_object* v_res_1956_; 
v___attrKind_boxed_1955_ = lean_unbox(v___attrKind_1951_);
v_res_1956_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(v___x_1944_, v___x_1945_, v___x_1946_, v___x_1947_, v___x_1948_, v_decl_1949_, v_stx_1950_, v___attrKind_boxed_1955_, v___y_1952_, v___y_1953_);
lean_dec(v___y_1953_);
lean_dec_ref(v___y_1952_);
lean_dec(v___x_1948_);
lean_dec(v___x_1947_);
return v_res_1956_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1957_; lean_object* v___x_1958_; lean_object* v___x_1959_; 
v___x_1957_ = lean_unsigned_to_nat(2535236365u);
v___x_1958_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__16_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_1959_ = l_Lean_Name_num___override(v___x_1958_, v___x_1957_);
return v___x_1959_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; 
v___x_1960_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__18_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_1961_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_);
v___x_1962_ = l_Lean_Name_str___override(v___x_1961_, v___x_1960_);
return v___x_1962_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; 
v___x_1963_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__20_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_1964_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_);
v___x_1965_ = l_Lean_Name_str___override(v___x_1964_, v___x_1963_);
return v___x_1965_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; 
v___x_1966_ = lean_unsigned_to_nat(2u);
v___x_1967_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_);
v___x_1968_ = l_Lean_Name_num___override(v___x_1967_, v___x_1966_);
return v___x_1968_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; 
v___x_1980_ = 2;
v___x_1981_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_));
v___x_1982_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_));
v___x_1983_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_);
v___x_1984_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_1984_, 0, v___x_1983_);
lean_ctor_set(v___x_1984_, 1, v___x_1982_);
lean_ctor_set(v___x_1984_, 2, v___x_1981_);
lean_ctor_set_uint8(v___x_1984_, sizeof(void*)*3, v___x_1980_);
return v___x_1984_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_1985_; lean_object* v___f_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; 
v___f_1985_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_));
v___f_1986_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_));
v___x_1987_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_);
v___x_1988_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1988_, 0, v___x_1987_);
lean_ctor_set(v___x_1988_, 1, v___f_1986_);
lean_ctor_set(v___x_1988_, 2, v___f_1985_);
return v___x_1988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1990_; lean_object* v___x_1991_; 
v___x_1990_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_);
v___x_1991_ = l_Lean_registerBuiltinAttribute(v___x_1990_);
return v___x_1991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2____boxed(lean_object* v_a_1992_){
_start:
{
lean_object* v_res_1993_; 
v_res_1993_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_();
return v_res_1993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter(lean_object* v_a_2029_, lean_object* v_a_2030_, lean_object* v_a_2031_, lean_object* v_a_2032_){
_start:
{
lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; 
v___x_2034_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter___closed__0));
v___x_2035_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Formatter_lmfdbIdNoAntiquot_formatter___boxed), 5, 0);
v___x_2036_ = l_Lean_PrettyPrinter_Formatter_orelse_formatter(v___x_2034_, v___x_2035_, v_a_2029_, v_a_2030_, v_a_2031_, v_a_2032_);
return v___x_2036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter___boxed(lean_object* v_a_2037_, lean_object* v_a_2038_, lean_object* v_a_2039_, lean_object* v_a_2040_, lean_object* v_a_2041_){
_start:
{
lean_object* v_res_2042_; 
v_res_2042_ = lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_formatter(v_a_2037_, v_a_2038_, v_a_2039_, v_a_2040_);
lean_dec(v_a_2040_);
lean_dec_ref(v_a_2039_);
lean_dec(v_a_2038_);
lean_dec_ref(v_a_2037_);
return v_res_2042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer(lean_object* v_a_2050_, lean_object* v_a_2051_, lean_object* v_a_2052_, lean_object* v_a_2053_){
_start:
{
lean_object* v___f_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; 
v___f_2055_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__0));
v___x_2056_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer___closed__0));
v___x_2057_ = l_Lean_PrettyPrinter_Parenthesizer_withAntiquot_parenthesizer(v___x_2056_, v___f_2055_, v_a_2050_, v_a_2051_, v_a_2052_, v_a_2053_);
return v___x_2057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer___boxed(lean_object* v_a_2058_, lean_object* v_a_2059_, lean_object* v_a_2060_, lean_object* v_a_2061_, lean_object* v_a_2062_){
_start:
{
lean_object* v_res_2063_; 
v_res_2063_ = lp_mathlib_Mathlib_CrossRef_lmfdbIdParser_parenthesizer(v_a_2058_, v_a_2059_, v_a_2060_, v_a_2061_);
lean_dec(v_a_2061_);
lean_dec_ref(v_a_2060_);
lean_dec(v_a_2059_);
lean_dec_ref(v_a_2058_);
return v_res_2063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(lean_object* v___x_2064_, lean_object* v___x_2065_, lean_object* v___x_2066_, lean_object* v___x_2067_, lean_object* v___x_2068_, lean_object* v_decl_2069_, lean_object* v_stx_2070_, uint8_t v___attrKind_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_){
_start:
{
lean_object* v_fst_2076_; lean_object* v_snd_2077_; lean_object* v___y_2078_; lean_object* v___y_2079_; lean_object* v___x_2096_; uint8_t v___x_2097_; 
v___x_2096_ = l_Lean_Name_mkStr3(v___x_2064_, v___x_2065_, v___x_2066_);
lean_inc(v_stx_2070_);
v___x_2097_ = l_Lean_Syntax_isOfKind(v_stx_2070_, v___x_2096_);
lean_dec(v___x_2096_);
if (v___x_2097_ == 0)
{
lean_object* v___x_2098_; lean_object* v_a_2099_; lean_object* v___x_2101_; uint8_t v_isShared_2102_; uint8_t v_isSharedCheck_2106_; 
lean_dec(v_stx_2070_);
lean_dec(v_decl_2069_);
v___x_2098_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_2099_ = lean_ctor_get(v___x_2098_, 0);
v_isSharedCheck_2106_ = !lean_is_exclusive(v___x_2098_);
if (v_isSharedCheck_2106_ == 0)
{
v___x_2101_ = v___x_2098_;
v_isShared_2102_ = v_isSharedCheck_2106_;
goto v_resetjp_2100_;
}
else
{
lean_inc(v_a_2099_);
lean_dec(v___x_2098_);
v___x_2101_ = lean_box(0);
v_isShared_2102_ = v_isSharedCheck_2106_;
goto v_resetjp_2100_;
}
v_resetjp_2100_:
{
lean_object* v___x_2104_; 
if (v_isShared_2102_ == 0)
{
v___x_2104_ = v___x_2101_;
goto v_reusejp_2103_;
}
else
{
lean_object* v_reuseFailAlloc_2105_; 
v_reuseFailAlloc_2105_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2105_, 0, v_a_2099_);
v___x_2104_ = v_reuseFailAlloc_2105_;
goto v_reusejp_2103_;
}
v_reusejp_2103_:
{
return v___x_2104_;
}
}
}
else
{
lean_object* v___x_2107_; lean_object* v_id_2108_; lean_object* v___x_2109_; uint8_t v___x_2110_; 
v___x_2107_ = lean_unsigned_to_nat(1u);
v_id_2108_ = l_Lean_Syntax_getArg(v_stx_2070_, v___x_2107_);
v___x_2109_ = l_Lean_Syntax_getArg(v_stx_2070_, v___x_2067_);
lean_dec(v_stx_2070_);
v___x_2110_ = l_Lean_Syntax_isNone(v___x_2109_);
if (v___x_2110_ == 0)
{
uint8_t v___x_2111_; 
lean_inc(v___x_2109_);
v___x_2111_ = l_Lean_Syntax_matchesNull(v___x_2109_, v___x_2107_);
if (v___x_2111_ == 0)
{
lean_object* v___x_2112_; lean_object* v_a_2113_; lean_object* v___x_2115_; uint8_t v_isShared_2116_; uint8_t v_isSharedCheck_2120_; 
lean_dec(v___x_2109_);
lean_dec(v_id_2108_);
lean_dec(v_decl_2069_);
v___x_2112_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_2113_ = lean_ctor_get(v___x_2112_, 0);
v_isSharedCheck_2120_ = !lean_is_exclusive(v___x_2112_);
if (v_isSharedCheck_2120_ == 0)
{
v___x_2115_ = v___x_2112_;
v_isShared_2116_ = v_isSharedCheck_2120_;
goto v_resetjp_2114_;
}
else
{
lean_inc(v_a_2113_);
lean_dec(v___x_2112_);
v___x_2115_ = lean_box(0);
v_isShared_2116_ = v_isSharedCheck_2120_;
goto v_resetjp_2114_;
}
v_resetjp_2114_:
{
lean_object* v___x_2118_; 
if (v_isShared_2116_ == 0)
{
v___x_2118_ = v___x_2115_;
goto v_reusejp_2117_;
}
else
{
lean_object* v_reuseFailAlloc_2119_; 
v_reuseFailAlloc_2119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2119_, 0, v_a_2113_);
v___x_2118_ = v_reuseFailAlloc_2119_;
goto v_reusejp_2117_;
}
v_reusejp_2117_:
{
return v___x_2118_;
}
}
}
else
{
lean_object* v_comment_2121_; lean_object* v___x_2122_; 
v_comment_2121_ = l_Lean_Syntax_getArg(v___x_2109_, v___x_2068_);
lean_dec(v___x_2109_);
v___x_2122_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2122_, 0, v_comment_2121_);
v_fst_2076_ = v_id_2108_;
v_snd_2077_ = v___x_2122_;
v___y_2078_ = v___y_2072_;
v___y_2079_ = v___y_2073_;
goto v___jp_2075_;
}
}
else
{
lean_object* v___x_2123_; 
lean_dec(v___x_2109_);
v___x_2123_ = lean_box(0);
v_fst_2076_ = v_id_2108_;
v_snd_2077_ = v___x_2123_;
v___y_2078_ = v___y_2072_;
v___y_2079_ = v___y_2073_;
goto v___jp_2075_;
}
}
v___jp_2075_:
{
lean_object* v___x_2080_; 
v___x_2080_ = lp_mathlib_Lean_TSyntax_getLmfdbId(v_fst_2076_, v___y_2078_, v___y_2079_);
lean_dec(v_fst_2076_);
if (lean_obj_tag(v___x_2080_) == 0)
{
lean_object* v_a_2081_; lean_object* v___x_2082_; 
v_a_2081_ = lean_ctor_get(v___x_2080_, 0);
lean_inc(v_a_2081_);
lean_dec_ref_known(v___x_2080_, 1);
v___x_2082_ = lean_box(2);
if (lean_obj_tag(v_snd_2077_) == 0)
{
lean_object* v___x_2083_; lean_object* v___x_2084_; 
v___x_2083_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___x_2084_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v___x_2082_, v_decl_2069_, v_a_2081_, v___x_2083_, v___y_2078_, v___y_2079_);
return v___x_2084_;
}
else
{
lean_object* v_val_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; 
v_val_2085_ = lean_ctor_get(v_snd_2077_, 0);
lean_inc(v_val_2085_);
lean_dec_ref_known(v_snd_2077_, 1);
v___x_2086_ = l_Lean_TSyntax_getString(v_val_2085_);
lean_dec(v_val_2085_);
v___x_2087_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v___x_2082_, v_decl_2069_, v_a_2081_, v___x_2086_, v___y_2078_, v___y_2079_);
return v___x_2087_;
}
}
else
{
lean_object* v_a_2088_; lean_object* v___x_2090_; uint8_t v_isShared_2091_; uint8_t v_isSharedCheck_2095_; 
lean_dec(v_snd_2077_);
lean_dec(v_decl_2069_);
v_a_2088_ = lean_ctor_get(v___x_2080_, 0);
v_isSharedCheck_2095_ = !lean_is_exclusive(v___x_2080_);
if (v_isSharedCheck_2095_ == 0)
{
v___x_2090_ = v___x_2080_;
v_isShared_2091_ = v_isSharedCheck_2095_;
goto v_resetjp_2089_;
}
else
{
lean_inc(v_a_2088_);
lean_dec(v___x_2080_);
v___x_2090_ = lean_box(0);
v_isShared_2091_ = v_isSharedCheck_2095_;
goto v_resetjp_2089_;
}
v_resetjp_2089_:
{
lean_object* v___x_2093_; 
if (v_isShared_2091_ == 0)
{
v___x_2093_ = v___x_2090_;
goto v_reusejp_2092_;
}
else
{
lean_object* v_reuseFailAlloc_2094_; 
v_reuseFailAlloc_2094_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2094_, 0, v_a_2088_);
v___x_2093_ = v_reuseFailAlloc_2094_;
goto v_reusejp_2092_;
}
v_reusejp_2092_:
{
return v___x_2093_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2____boxed(lean_object* v___x_2124_, lean_object* v___x_2125_, lean_object* v___x_2126_, lean_object* v___x_2127_, lean_object* v___x_2128_, lean_object* v_decl_2129_, lean_object* v_stx_2130_, lean_object* v___attrKind_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_){
_start:
{
uint8_t v___attrKind_boxed_2135_; lean_object* v_res_2136_; 
v___attrKind_boxed_2135_ = lean_unbox(v___attrKind_2131_);
v_res_2136_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(v___x_2124_, v___x_2125_, v___x_2126_, v___x_2127_, v___x_2128_, v_decl_2129_, v_stx_2130_, v___attrKind_boxed_2135_, v___y_2132_, v___y_2133_);
lean_dec(v___y_2133_);
lean_dec_ref(v___y_2132_);
lean_dec(v___x_2128_);
lean_dec(v___x_2127_);
return v_res_2136_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2137_; lean_object* v___x_2138_; lean_object* v___x_2139_; 
v___x_2137_ = lean_unsigned_to_nat(3593266836u);
v___x_2138_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__16_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_2139_ = l_Lean_Name_num___override(v___x_2138_, v___x_2137_);
return v___x_2139_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; 
v___x_2140_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__18_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_2141_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_);
v___x_2142_ = l_Lean_Name_str___override(v___x_2141_, v___x_2140_);
return v___x_2142_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; 
v___x_2143_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__20_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_2144_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_);
v___x_2145_ = l_Lean_Name_str___override(v___x_2144_, v___x_2143_);
return v___x_2145_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; 
v___x_2146_ = lean_unsigned_to_nat(2u);
v___x_2147_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_);
v___x_2148_ = l_Lean_Name_num___override(v___x_2147_, v___x_2146_);
return v___x_2148_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; 
v___x_2160_ = 2;
v___x_2161_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_));
v___x_2162_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_));
v___x_2163_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_);
v___x_2164_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_2164_, 0, v___x_2163_);
lean_ctor_set(v___x_2164_, 1, v___x_2162_);
lean_ctor_set(v___x_2164_, 2, v___x_2161_);
lean_ctor_set_uint8(v___x_2164_, sizeof(void*)*3, v___x_2160_);
return v___x_2164_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_2165_; lean_object* v___f_2166_; lean_object* v___x_2167_; lean_object* v___x_2168_; 
v___f_2165_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_));
v___f_2166_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_));
v___x_2167_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_);
v___x_2168_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2168_, 0, v___x_2167_);
lean_ctor_set(v___x_2168_, 1, v___f_2166_);
lean_ctor_set(v___x_2168_, 2, v___f_2165_);
return v___x_2168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2170_; lean_object* v___x_2171_; 
v___x_2170_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_);
v___x_2171_ = l_Lean_registerBuiltinAttribute(v___x_2170_);
return v___x_2171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2____boxed(lean_object* v_a_2172_){
_start:
{
lean_object* v_res_2173_; 
v_res_2173_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_();
return v_res_2173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_getPiBaseTopic_x3f(lean_object* v_x_2189_){
_start:
{
lean_object* v___x_2190_; uint8_t v___x_2191_; 
v___x_2190_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseTopic___closed__1));
v___x_2191_ = l_Lean_Syntax_isOfKind(v_x_2189_, v___x_2190_);
if (v___x_2191_ == 0)
{
lean_object* v___x_2192_; 
v___x_2192_ = lean_box(0);
return v___x_2192_;
}
else
{
lean_object* v___x_2193_; 
v___x_2193_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_getPiBaseTopic_x3f___closed__0));
return v___x_2193_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter(lean_object* v_a_2234_, lean_object* v_a_2235_, lean_object* v_a_2236_, lean_object* v_a_2237_){
_start:
{
lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; 
v___x_2239_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter___closed__0));
v___x_2240_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Formatter_pibaseIdNoAntiquot_formatter___boxed), 5, 0);
v___x_2241_ = l_Lean_PrettyPrinter_Formatter_orelse_formatter(v___x_2239_, v___x_2240_, v_a_2234_, v_a_2235_, v_a_2236_, v_a_2237_);
return v___x_2241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter___boxed(lean_object* v_a_2242_, lean_object* v_a_2243_, lean_object* v_a_2244_, lean_object* v_a_2245_, lean_object* v_a_2246_){
_start:
{
lean_object* v_res_2247_; 
v_res_2247_ = lp_mathlib_Mathlib_CrossRef_pibaseIdParser_formatter(v_a_2242_, v_a_2243_, v_a_2244_, v_a_2245_);
lean_dec(v_a_2245_);
lean_dec_ref(v_a_2244_);
lean_dec(v_a_2243_);
lean_dec_ref(v_a_2242_);
return v_res_2247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer(lean_object* v_a_2255_, lean_object* v_a_2256_, lean_object* v_a_2257_, lean_object* v_a_2258_){
_start:
{
lean_object* v___f_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; 
v___f_2260_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__0));
v___x_2261_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer___closed__0));
v___x_2262_ = l_Lean_PrettyPrinter_Parenthesizer_withAntiquot_parenthesizer(v___x_2261_, v___f_2260_, v_a_2255_, v_a_2256_, v_a_2257_, v_a_2258_);
return v___x_2262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer___boxed(lean_object* v_a_2263_, lean_object* v_a_2264_, lean_object* v_a_2265_, lean_object* v_a_2266_, lean_object* v_a_2267_){
_start:
{
lean_object* v_res_2268_; 
v_res_2268_ = lp_mathlib_Mathlib_CrossRef_pibaseIdParser_parenthesizer(v_a_2263_, v_a_2264_, v_a_2265_, v_a_2266_);
lean_dec(v_a_2266_);
lean_dec_ref(v_a_2265_);
lean_dec(v_a_2264_);
lean_dec_ref(v_a_2263_);
return v_res_2268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(lean_object* v___x_2269_, lean_object* v___x_2270_, lean_object* v___x_2271_, lean_object* v___x_2272_, lean_object* v___x_2273_, lean_object* v_decl_2274_, lean_object* v_stx_2275_, uint8_t v___attrKind_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_){
_start:
{
lean_object* v_fst_2281_; lean_object* v_fst_2282_; lean_object* v_snd_2283_; lean_object* v___y_2284_; lean_object* v___y_2285_; lean_object* v___x_2302_; uint8_t v___x_2303_; 
v___x_2302_ = l_Lean_Name_mkStr3(v___x_2269_, v___x_2270_, v___x_2271_);
lean_inc(v_stx_2275_);
v___x_2303_ = l_Lean_Syntax_isOfKind(v_stx_2275_, v___x_2302_);
lean_dec(v___x_2302_);
if (v___x_2303_ == 0)
{
lean_object* v___x_2304_; lean_object* v_a_2305_; lean_object* v___x_2307_; uint8_t v_isShared_2308_; uint8_t v_isSharedCheck_2312_; 
lean_dec(v_stx_2275_);
lean_dec(v_decl_2274_);
v___x_2304_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_2305_ = lean_ctor_get(v___x_2304_, 0);
v_isSharedCheck_2312_ = !lean_is_exclusive(v___x_2304_);
if (v_isSharedCheck_2312_ == 0)
{
v___x_2307_ = v___x_2304_;
v_isShared_2308_ = v_isSharedCheck_2312_;
goto v_resetjp_2306_;
}
else
{
lean_inc(v_a_2305_);
lean_dec(v___x_2304_);
v___x_2307_ = lean_box(0);
v_isShared_2308_ = v_isSharedCheck_2312_;
goto v_resetjp_2306_;
}
v_resetjp_2306_:
{
lean_object* v___x_2310_; 
if (v_isShared_2308_ == 0)
{
v___x_2310_ = v___x_2307_;
goto v_reusejp_2309_;
}
else
{
lean_object* v_reuseFailAlloc_2311_; 
v_reuseFailAlloc_2311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2311_, 0, v_a_2305_);
v___x_2310_ = v_reuseFailAlloc_2311_;
goto v_reusejp_2309_;
}
v_reusejp_2309_:
{
return v___x_2310_;
}
}
}
else
{
lean_object* v___x_2313_; lean_object* v_topic_2314_; lean_object* v_id_2315_; lean_object* v_comment_2317_; lean_object* v___y_2318_; lean_object* v___y_2319_; lean_object* v___x_2331_; lean_object* v___x_2332_; uint8_t v___x_2333_; 
v___x_2313_ = lean_unsigned_to_nat(1u);
v_topic_2314_ = l_Lean_Syntax_getArg(v_stx_2275_, v___x_2313_);
v_id_2315_ = l_Lean_Syntax_getArg(v_stx_2275_, v___x_2272_);
v___x_2331_ = lean_unsigned_to_nat(3u);
v___x_2332_ = l_Lean_Syntax_getArg(v_stx_2275_, v___x_2331_);
lean_dec(v_stx_2275_);
v___x_2333_ = l_Lean_Syntax_isNone(v___x_2332_);
if (v___x_2333_ == 0)
{
uint8_t v___x_2334_; 
lean_inc(v___x_2332_);
v___x_2334_ = l_Lean_Syntax_matchesNull(v___x_2332_, v___x_2313_);
if (v___x_2334_ == 0)
{
lean_object* v___x_2335_; lean_object* v_a_2336_; lean_object* v___x_2338_; uint8_t v_isShared_2339_; uint8_t v_isSharedCheck_2343_; 
lean_dec(v___x_2332_);
lean_dec(v_id_2315_);
lean_dec(v_topic_2314_);
lean_dec(v_decl_2274_);
v___x_2335_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_2336_ = lean_ctor_get(v___x_2335_, 0);
v_isSharedCheck_2343_ = !lean_is_exclusive(v___x_2335_);
if (v_isSharedCheck_2343_ == 0)
{
v___x_2338_ = v___x_2335_;
v_isShared_2339_ = v_isSharedCheck_2343_;
goto v_resetjp_2337_;
}
else
{
lean_inc(v_a_2336_);
lean_dec(v___x_2335_);
v___x_2338_ = lean_box(0);
v_isShared_2339_ = v_isSharedCheck_2343_;
goto v_resetjp_2337_;
}
v_resetjp_2337_:
{
lean_object* v___x_2341_; 
if (v_isShared_2339_ == 0)
{
v___x_2341_ = v___x_2338_;
goto v_reusejp_2340_;
}
else
{
lean_object* v_reuseFailAlloc_2342_; 
v_reuseFailAlloc_2342_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2342_, 0, v_a_2336_);
v___x_2341_ = v_reuseFailAlloc_2342_;
goto v_reusejp_2340_;
}
v_reusejp_2340_:
{
return v___x_2341_;
}
}
}
else
{
lean_object* v_comment_2344_; lean_object* v___x_2345_; 
v_comment_2344_ = l_Lean_Syntax_getArg(v___x_2332_, v___x_2273_);
lean_dec(v___x_2332_);
v___x_2345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2345_, 0, v_comment_2344_);
v_comment_2317_ = v___x_2345_;
v___y_2318_ = v___y_2277_;
v___y_2319_ = v___y_2278_;
goto v___jp_2316_;
}
}
else
{
lean_object* v___x_2346_; 
lean_dec(v___x_2332_);
v___x_2346_ = lean_box(0);
v_comment_2317_ = v___x_2346_;
v___y_2318_ = v___y_2277_;
v___y_2319_ = v___y_2278_;
goto v___jp_2316_;
}
v___jp_2316_:
{
lean_object* v___x_2320_; 
v___x_2320_ = lp_mathlib_Mathlib_CrossRef_getPiBaseTopic_x3f(v_topic_2314_);
if (lean_obj_tag(v___x_2320_) == 1)
{
lean_object* v_val_2321_; 
v_val_2321_ = lean_ctor_get(v___x_2320_, 0);
lean_inc(v_val_2321_);
lean_dec_ref_known(v___x_2320_, 1);
v_fst_2281_ = v_id_2315_;
v_fst_2282_ = v_val_2321_;
v_snd_2283_ = v_comment_2317_;
v___y_2284_ = v___y_2318_;
v___y_2285_ = v___y_2319_;
goto v___jp_2280_;
}
else
{
lean_object* v___x_2322_; lean_object* v_a_2323_; lean_object* v___x_2325_; uint8_t v_isShared_2326_; uint8_t v_isSharedCheck_2330_; 
lean_dec(v___x_2320_);
lean_dec(v_comment_2317_);
lean_dec(v_id_2315_);
lean_dec(v_decl_2274_);
v___x_2322_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_2323_ = lean_ctor_get(v___x_2322_, 0);
v_isSharedCheck_2330_ = !lean_is_exclusive(v___x_2322_);
if (v_isSharedCheck_2330_ == 0)
{
v___x_2325_ = v___x_2322_;
v_isShared_2326_ = v_isSharedCheck_2330_;
goto v_resetjp_2324_;
}
else
{
lean_inc(v_a_2323_);
lean_dec(v___x_2322_);
v___x_2325_ = lean_box(0);
v_isShared_2326_ = v_isSharedCheck_2330_;
goto v_resetjp_2324_;
}
v_resetjp_2324_:
{
lean_object* v___x_2328_; 
if (v_isShared_2326_ == 0)
{
v___x_2328_ = v___x_2325_;
goto v_reusejp_2327_;
}
else
{
lean_object* v_reuseFailAlloc_2329_; 
v_reuseFailAlloc_2329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2329_, 0, v_a_2323_);
v___x_2328_ = v_reuseFailAlloc_2329_;
goto v_reusejp_2327_;
}
v_reusejp_2327_:
{
return v___x_2328_;
}
}
}
}
}
v___jp_2280_:
{
lean_object* v___x_2286_; 
v___x_2286_ = lp_mathlib_Lean_TSyntax_getPibaseId(v_fst_2281_, v___y_2284_, v___y_2285_);
lean_dec(v_fst_2281_);
if (lean_obj_tag(v___x_2286_) == 0)
{
lean_object* v_a_2287_; lean_object* v___x_2288_; 
v_a_2287_ = lean_ctor_get(v___x_2286_, 0);
lean_inc(v_a_2287_);
lean_dec_ref_known(v___x_2286_, 1);
v___x_2288_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2288_, 0, v_fst_2282_);
if (lean_obj_tag(v_snd_2283_) == 0)
{
lean_object* v___x_2289_; lean_object* v___x_2290_; 
v___x_2289_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___x_2290_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v___x_2288_, v_decl_2274_, v_a_2287_, v___x_2289_, v___y_2284_, v___y_2285_);
return v___x_2290_;
}
else
{
lean_object* v_val_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; 
v_val_2291_ = lean_ctor_get(v_snd_2283_, 0);
lean_inc(v_val_2291_);
lean_dec_ref_known(v_snd_2283_, 1);
v___x_2292_ = l_Lean_TSyntax_getString(v_val_2291_);
lean_dec(v_val_2291_);
v___x_2293_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v___x_2288_, v_decl_2274_, v_a_2287_, v___x_2292_, v___y_2284_, v___y_2285_);
return v___x_2293_;
}
}
else
{
lean_object* v_a_2294_; lean_object* v___x_2296_; uint8_t v_isShared_2297_; uint8_t v_isSharedCheck_2301_; 
lean_dec(v_snd_2283_);
lean_dec(v_decl_2274_);
v_a_2294_ = lean_ctor_get(v___x_2286_, 0);
v_isSharedCheck_2301_ = !lean_is_exclusive(v___x_2286_);
if (v_isSharedCheck_2301_ == 0)
{
v___x_2296_ = v___x_2286_;
v_isShared_2297_ = v_isSharedCheck_2301_;
goto v_resetjp_2295_;
}
else
{
lean_inc(v_a_2294_);
lean_dec(v___x_2286_);
v___x_2296_ = lean_box(0);
v_isShared_2297_ = v_isSharedCheck_2301_;
goto v_resetjp_2295_;
}
v_resetjp_2295_:
{
lean_object* v___x_2299_; 
if (v_isShared_2297_ == 0)
{
v___x_2299_ = v___x_2296_;
goto v_reusejp_2298_;
}
else
{
lean_object* v_reuseFailAlloc_2300_; 
v_reuseFailAlloc_2300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2300_, 0, v_a_2294_);
v___x_2299_ = v_reuseFailAlloc_2300_;
goto v_reusejp_2298_;
}
v_reusejp_2298_:
{
return v___x_2299_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2____boxed(lean_object* v___x_2347_, lean_object* v___x_2348_, lean_object* v___x_2349_, lean_object* v___x_2350_, lean_object* v___x_2351_, lean_object* v_decl_2352_, lean_object* v_stx_2353_, lean_object* v___attrKind_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_, lean_object* v___y_2357_){
_start:
{
uint8_t v___attrKind_boxed_2358_; lean_object* v_res_2359_; 
v___attrKind_boxed_2358_ = lean_unbox(v___attrKind_2354_);
v_res_2359_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(v___x_2347_, v___x_2348_, v___x_2349_, v___x_2350_, v___x_2351_, v_decl_2352_, v_stx_2353_, v___attrKind_boxed_2358_, v___y_2355_, v___y_2356_);
lean_dec(v___y_2356_);
lean_dec_ref(v___y_2355_);
lean_dec(v___x_2351_);
lean_dec(v___x_2350_);
return v_res_2359_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; 
v___x_2360_ = lean_unsigned_to_nat(2958650249u);
v___x_2361_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__16_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_2362_ = l_Lean_Name_num___override(v___x_2361_, v___x_2360_);
return v___x_2362_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; 
v___x_2363_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__18_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_2364_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__0_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_);
v___x_2365_ = l_Lean_Name_str___override(v___x_2364_, v___x_2363_);
return v___x_2365_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; 
v___x_2366_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__20_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_));
v___x_2367_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__1_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_);
v___x_2368_ = l_Lean_Name_str___override(v___x_2367_, v___x_2366_);
return v___x_2368_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; 
v___x_2369_ = lean_unsigned_to_nat(2u);
v___x_2370_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__2_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_);
v___x_2371_ = l_Lean_Name_num___override(v___x_2370_, v___x_2369_);
return v___x_2371_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; 
v___x_2383_ = 2;
v___x_2384_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__7_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_));
v___x_2385_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__5_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_));
v___x_2386_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__3_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_);
v___x_2387_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_2387_, 0, v___x_2386_);
lean_ctor_set(v___x_2387_, 1, v___x_2385_);
lean_ctor_set(v___x_2387_, 2, v___x_2384_);
lean_ctor_set_uint8(v___x_2387_, sizeof(void*)*3, v___x_2383_);
return v___x_2387_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_2388_; lean_object* v___f_2389_; lean_object* v___x_2390_; lean_object* v___x_2391_; 
v___f_2388_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__6_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_));
v___f_2389_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__4_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_));
v___x_2390_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__8_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_);
v___x_2391_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2391_, 0, v___x_2390_);
lean_ctor_set(v___x_2391_, 1, v___f_2389_);
lean_ctor_set(v___x_2391_, 2, v___f_2388_);
return v___x_2391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2393_; lean_object* v___x_2394_; 
v___x_2393_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_);
v___x_2394_ = l_Lean_registerBuiltinAttribute(v___x_2393_);
return v___x_2394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2____boxed(lean_object* v_a_2395_){
_start:
{
lean_object* v_res_2396_; 
v_res_2396_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_();
return v_res_2396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter(lean_object* v_a_2432_, lean_object* v_a_2433_, lean_object* v_a_2434_, lean_object* v_a_2435_){
_start:
{
lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; 
v___x_2437_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter___closed__0));
v___x_2438_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Formatter_dlmfIdNoAntiquot_formatter___boxed), 5, 0);
v___x_2439_ = l_Lean_PrettyPrinter_Formatter_orelse_formatter(v___x_2437_, v___x_2438_, v_a_2432_, v_a_2433_, v_a_2434_, v_a_2435_);
return v___x_2439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter___boxed(lean_object* v_a_2440_, lean_object* v_a_2441_, lean_object* v_a_2442_, lean_object* v_a_2443_, lean_object* v_a_2444_){
_start:
{
lean_object* v_res_2445_; 
v_res_2445_ = lp_mathlib_Mathlib_CrossRef_dlmfIdParser_formatter(v_a_2440_, v_a_2441_, v_a_2442_, v_a_2443_);
lean_dec(v_a_2443_);
lean_dec_ref(v_a_2442_);
lean_dec(v_a_2441_);
lean_dec_ref(v_a_2440_);
return v_res_2445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer(lean_object* v_a_2453_, lean_object* v_a_2454_, lean_object* v_a_2455_, lean_object* v_a_2456_){
_start:
{
lean_object* v___f_2458_; lean_object* v___x_2459_; lean_object* v___x_2460_; 
v___f_2458_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTagParser_parenthesizer___closed__0));
v___x_2459_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer___closed__0));
v___x_2460_ = l_Lean_PrettyPrinter_Parenthesizer_withAntiquot_parenthesizer(v___x_2459_, v___f_2458_, v_a_2453_, v_a_2454_, v_a_2455_, v_a_2456_);
return v___x_2460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer___boxed(lean_object* v_a_2461_, lean_object* v_a_2462_, lean_object* v_a_2463_, lean_object* v_a_2464_, lean_object* v_a_2465_){
_start:
{
lean_object* v_res_2466_; 
v_res_2466_ = lp_mathlib_Mathlib_CrossRef_dlmfIdParser_parenthesizer(v_a_2461_, v_a_2462_, v_a_2463_, v_a_2464_);
lean_dec(v_a_2464_);
lean_dec_ref(v_a_2463_);
lean_dec(v_a_2462_);
lean_dec_ref(v_a_2461_);
return v_res_2466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_(lean_object* v___x_2467_, lean_object* v___x_2468_, lean_object* v___x_2469_, lean_object* v___x_2470_, lean_object* v___x_2471_, lean_object* v_decl_2472_, lean_object* v_stx_2473_, uint8_t v___attrKind_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_){
_start:
{
lean_object* v_fst_2479_; lean_object* v_snd_2480_; lean_object* v___y_2481_; lean_object* v___y_2482_; lean_object* v___x_2499_; uint8_t v___x_2500_; 
v___x_2499_ = l_Lean_Name_mkStr3(v___x_2467_, v___x_2468_, v___x_2469_);
lean_inc(v_stx_2473_);
v___x_2500_ = l_Lean_Syntax_isOfKind(v_stx_2473_, v___x_2499_);
lean_dec(v___x_2499_);
if (v___x_2500_ == 0)
{
lean_object* v___x_2501_; lean_object* v_a_2502_; lean_object* v___x_2504_; uint8_t v_isShared_2505_; uint8_t v_isSharedCheck_2509_; 
lean_dec(v_stx_2473_);
lean_dec(v_decl_2472_);
v___x_2501_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_2502_ = lean_ctor_get(v___x_2501_, 0);
v_isSharedCheck_2509_ = !lean_is_exclusive(v___x_2501_);
if (v_isSharedCheck_2509_ == 0)
{
v___x_2504_ = v___x_2501_;
v_isShared_2505_ = v_isSharedCheck_2509_;
goto v_resetjp_2503_;
}
else
{
lean_inc(v_a_2502_);
lean_dec(v___x_2501_);
v___x_2504_ = lean_box(0);
v_isShared_2505_ = v_isSharedCheck_2509_;
goto v_resetjp_2503_;
}
v_resetjp_2503_:
{
lean_object* v___x_2507_; 
if (v_isShared_2505_ == 0)
{
v___x_2507_ = v___x_2504_;
goto v_reusejp_2506_;
}
else
{
lean_object* v_reuseFailAlloc_2508_; 
v_reuseFailAlloc_2508_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2508_, 0, v_a_2502_);
v___x_2507_ = v_reuseFailAlloc_2508_;
goto v_reusejp_2506_;
}
v_reusejp_2506_:
{
return v___x_2507_;
}
}
}
else
{
lean_object* v___x_2510_; lean_object* v_id_2511_; lean_object* v___x_2512_; uint8_t v___x_2513_; 
v___x_2510_ = lean_unsigned_to_nat(1u);
v_id_2511_ = l_Lean_Syntax_getArg(v_stx_2473_, v___x_2510_);
v___x_2512_ = l_Lean_Syntax_getArg(v_stx_2473_, v___x_2470_);
lean_dec(v_stx_2473_);
v___x_2513_ = l_Lean_Syntax_isNone(v___x_2512_);
if (v___x_2513_ == 0)
{
uint8_t v___x_2514_; 
lean_inc(v___x_2512_);
v___x_2514_ = l_Lean_Syntax_matchesNull(v___x_2512_, v___x_2510_);
if (v___x_2514_ == 0)
{
lean_object* v___x_2515_; lean_object* v_a_2516_; lean_object* v___x_2518_; uint8_t v_isShared_2519_; uint8_t v_isSharedCheck_2523_; 
lean_dec(v___x_2512_);
lean_dec(v_id_2511_);
lean_dec(v_decl_2472_);
v___x_2515_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg();
v_a_2516_ = lean_ctor_get(v___x_2515_, 0);
v_isSharedCheck_2523_ = !lean_is_exclusive(v___x_2515_);
if (v_isSharedCheck_2523_ == 0)
{
v___x_2518_ = v___x_2515_;
v_isShared_2519_ = v_isSharedCheck_2523_;
goto v_resetjp_2517_;
}
else
{
lean_inc(v_a_2516_);
lean_dec(v___x_2515_);
v___x_2518_ = lean_box(0);
v_isShared_2519_ = v_isSharedCheck_2523_;
goto v_resetjp_2517_;
}
v_resetjp_2517_:
{
lean_object* v___x_2521_; 
if (v_isShared_2519_ == 0)
{
v___x_2521_ = v___x_2518_;
goto v_reusejp_2520_;
}
else
{
lean_object* v_reuseFailAlloc_2522_; 
v_reuseFailAlloc_2522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2522_, 0, v_a_2516_);
v___x_2521_ = v_reuseFailAlloc_2522_;
goto v_reusejp_2520_;
}
v_reusejp_2520_:
{
return v___x_2521_;
}
}
}
else
{
lean_object* v_comment_2524_; lean_object* v___x_2525_; 
v_comment_2524_ = l_Lean_Syntax_getArg(v___x_2512_, v___x_2471_);
lean_dec(v___x_2512_);
v___x_2525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2525_, 0, v_comment_2524_);
v_fst_2479_ = v_id_2511_;
v_snd_2480_ = v___x_2525_;
v___y_2481_ = v___y_2475_;
v___y_2482_ = v___y_2476_;
goto v___jp_2478_;
}
}
else
{
lean_object* v___x_2526_; 
lean_dec(v___x_2512_);
v___x_2526_ = lean_box(0);
v_fst_2479_ = v_id_2511_;
v_snd_2480_ = v___x_2526_;
v___y_2481_ = v___y_2475_;
v___y_2482_ = v___y_2476_;
goto v___jp_2478_;
}
}
v___jp_2478_:
{
lean_object* v___x_2483_; 
v___x_2483_ = lp_mathlib_Lean_TSyntax_getDlmfId(v_fst_2479_, v___y_2481_, v___y_2482_);
lean_dec(v_fst_2479_);
if (lean_obj_tag(v___x_2483_) == 0)
{
lean_object* v_a_2484_; lean_object* v___x_2485_; 
v_a_2484_ = lean_ctor_get(v___x_2483_, 0);
lean_inc(v_a_2484_);
lean_dec_ref_known(v___x_2483_, 1);
v___x_2485_ = lean_box(0);
if (lean_obj_tag(v_snd_2480_) == 0)
{
lean_object* v___x_2486_; lean_object* v___x_2487_; 
v___x_2486_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___x_2487_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v___x_2485_, v_decl_2472_, v_a_2484_, v___x_2486_, v___y_2481_, v___y_2482_);
return v___x_2487_;
}
else
{
lean_object* v_val_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; 
v_val_2488_ = lean_ctor_get(v_snd_2480_, 0);
lean_inc(v_val_2488_);
lean_dec_ref_known(v_snd_2480_, 1);
v___x_2489_ = l_Lean_TSyntax_getString(v_val_2488_);
lean_dec(v_val_2488_);
v___x_2490_ = lp_mathlib_Mathlib_CrossRef_addCrossRefDoc(v___x_2485_, v_decl_2472_, v_a_2484_, v___x_2489_, v___y_2481_, v___y_2482_);
return v___x_2490_;
}
}
else
{
lean_object* v_a_2491_; lean_object* v___x_2493_; uint8_t v_isShared_2494_; uint8_t v_isSharedCheck_2498_; 
lean_dec(v_snd_2480_);
lean_dec(v_decl_2472_);
v_a_2491_ = lean_ctor_get(v___x_2483_, 0);
v_isSharedCheck_2498_ = !lean_is_exclusive(v___x_2483_);
if (v_isSharedCheck_2498_ == 0)
{
v___x_2493_ = v___x_2483_;
v_isShared_2494_ = v_isSharedCheck_2498_;
goto v_resetjp_2492_;
}
else
{
lean_inc(v_a_2491_);
lean_dec(v___x_2483_);
v___x_2493_ = lean_box(0);
v_isShared_2494_ = v_isSharedCheck_2498_;
goto v_resetjp_2492_;
}
v_resetjp_2492_:
{
lean_object* v___x_2496_; 
if (v_isShared_2494_ == 0)
{
v___x_2496_ = v___x_2493_;
goto v_reusejp_2495_;
}
else
{
lean_object* v_reuseFailAlloc_2497_; 
v_reuseFailAlloc_2497_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2497_, 0, v_a_2491_);
v___x_2496_ = v_reuseFailAlloc_2497_;
goto v_reusejp_2495_;
}
v_reusejp_2495_:
{
return v___x_2496_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2____boxed(lean_object* v___x_2527_, lean_object* v___x_2528_, lean_object* v___x_2529_, lean_object* v___x_2530_, lean_object* v___x_2531_, lean_object* v_decl_2532_, lean_object* v_stx_2533_, lean_object* v___attrKind_2534_, lean_object* v___y_2535_, lean_object* v___y_2536_, lean_object* v___y_2537_){
_start:
{
uint8_t v___attrKind_boxed_2538_; lean_object* v_res_2539_; 
v___attrKind_boxed_2538_ = lean_unbox(v___attrKind_2534_);
v_res_2539_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___lam__0_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_(v___x_2527_, v___x_2528_, v___x_2529_, v___x_2530_, v___x_2531_, v_decl_2532_, v_stx_2533_, v___attrKind_boxed_2538_, v___y_2535_, v___y_2536_);
lean_dec(v___y_2536_);
lean_dec_ref(v___y_2535_);
lean_dec(v___x_2531_);
lean_dec(v___x_2530_);
return v_res_2539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_2573_; lean_object* v___x_2574_; 
v___x_2573_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn___closed__9_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_));
v___x_2574_ = l_Lean_registerBuiltinAttribute(v___x_2573_);
return v___x_2574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2____boxed(lean_object* v_a_2575_){
_start:
{
lean_object* v_res_2576_; 
v_res_2576_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_();
return v_res_2576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__1(lean_object* v_as_2577_, size_t v_i_2578_, size_t v_stop_2579_, lean_object* v_b_2580_){
_start:
{
uint8_t v___x_2581_; 
v___x_2581_ = lean_usize_dec_eq(v_i_2578_, v_stop_2579_);
if (v___x_2581_ == 0)
{
lean_object* v___x_2582_; lean_object* v___x_2583_; size_t v___x_2584_; size_t v___x_2585_; 
v___x_2582_ = lean_array_uget_borrowed(v_as_2577_, v_i_2578_);
v___x_2583_ = l_Array_append___redArg(v_b_2580_, v___x_2582_);
v___x_2584_ = ((size_t)1ULL);
v___x_2585_ = lean_usize_add(v_i_2578_, v___x_2584_);
v_i_2578_ = v___x_2585_;
v_b_2580_ = v___x_2583_;
goto _start;
}
else
{
return v_b_2580_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__1___boxed(lean_object* v_as_2587_, lean_object* v_i_2588_, lean_object* v_stop_2589_, lean_object* v_b_2590_){
_start:
{
size_t v_i_boxed_2591_; size_t v_stop_boxed_2592_; lean_object* v_res_2593_; 
v_i_boxed_2591_ = lean_unbox_usize(v_i_2588_);
lean_dec(v_i_2588_);
v_stop_boxed_2592_ = lean_unbox_usize(v_stop_2589_);
lean_dec(v_stop_2589_);
v_res_2593_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__1(v_as_2587_, v_i_boxed_2591_, v_stop_boxed_2592_, v_b_2590_);
lean_dec_ref(v_as_2587_);
return v_res_2593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___redArg(lean_object* v_hi_2594_, lean_object* v_pivot_2595_, lean_object* v_as_2596_, lean_object* v_i_2597_, lean_object* v_k_2598_){
_start:
{
uint8_t v___x_2599_; 
v___x_2599_ = lean_nat_dec_lt(v_k_2598_, v_hi_2594_);
if (v___x_2599_ == 0)
{
lean_object* v___x_2600_; lean_object* v___x_2601_; 
lean_dec(v_k_2598_);
v___x_2600_ = lean_array_fswap(v_as_2596_, v_i_2597_, v_hi_2594_);
v___x_2601_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2601_, 0, v_i_2597_);
lean_ctor_set(v___x_2601_, 1, v___x_2600_);
return v___x_2601_;
}
else
{
lean_object* v___x_2602_; lean_object* v_tag_2603_; lean_object* v_tag_2604_; uint8_t v___x_2605_; 
v___x_2602_ = lean_array_fget_borrowed(v_as_2596_, v_k_2598_);
v_tag_2603_ = lean_ctor_get(v___x_2602_, 2);
v_tag_2604_ = lean_ctor_get(v_pivot_2595_, 2);
v___x_2605_ = lean_string_dec_lt(v_tag_2603_, v_tag_2604_);
if (v___x_2605_ == 0)
{
lean_object* v___x_2606_; lean_object* v___x_2607_; 
v___x_2606_ = lean_unsigned_to_nat(1u);
v___x_2607_ = lean_nat_add(v_k_2598_, v___x_2606_);
lean_dec(v_k_2598_);
v_k_2598_ = v___x_2607_;
goto _start;
}
else
{
lean_object* v___x_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; 
v___x_2609_ = lean_array_fswap(v_as_2596_, v_i_2597_, v_k_2598_);
v___x_2610_ = lean_unsigned_to_nat(1u);
v___x_2611_ = lean_nat_add(v_i_2597_, v___x_2610_);
lean_dec(v_i_2597_);
v___x_2612_ = lean_nat_add(v_k_2598_, v___x_2610_);
lean_dec(v_k_2598_);
v_as_2596_ = v___x_2609_;
v_i_2597_ = v___x_2611_;
v_k_2598_ = v___x_2612_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___redArg___boxed(lean_object* v_hi_2614_, lean_object* v_pivot_2615_, lean_object* v_as_2616_, lean_object* v_i_2617_, lean_object* v_k_2618_){
_start:
{
lean_object* v_res_2619_; 
v_res_2619_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___redArg(v_hi_2614_, v_pivot_2615_, v_as_2616_, v_i_2617_, v_k_2618_);
lean_dec_ref(v_pivot_2615_);
lean_dec(v_hi_2614_);
return v_res_2619_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___lam__0(lean_object* v_x1_2620_, lean_object* v_x2_2621_){
_start:
{
lean_object* v_tag_2622_; lean_object* v_tag_2623_; uint8_t v___x_2624_; 
v_tag_2622_ = lean_ctor_get(v_x1_2620_, 2);
v_tag_2623_ = lean_ctor_get(v_x2_2621_, 2);
v___x_2624_ = lean_string_dec_lt(v_tag_2622_, v_tag_2623_);
return v___x_2624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___lam__0___boxed(lean_object* v_x1_2625_, lean_object* v_x2_2626_){
_start:
{
uint8_t v_res_2627_; lean_object* v_r_2628_; 
v_res_2627_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___lam__0(v_x1_2625_, v_x2_2626_);
lean_dec_ref(v_x2_2626_);
lean_dec_ref(v_x1_2625_);
v_r_2628_ = lean_box(v_res_2627_);
return v_r_2628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg(lean_object* v_n_2629_, lean_object* v_as_2630_, lean_object* v_lo_2631_, lean_object* v_hi_2632_){
_start:
{
lean_object* v___y_2634_; uint8_t v___x_2644_; 
v___x_2644_ = lean_nat_dec_lt(v_lo_2631_, v_hi_2632_);
if (v___x_2644_ == 0)
{
lean_dec(v_lo_2631_);
return v_as_2630_;
}
else
{
lean_object* v___x_2645_; lean_object* v___x_2646_; lean_object* v_mid_2647_; lean_object* v___y_2649_; lean_object* v___y_2655_; lean_object* v___x_2660_; lean_object* v___x_2661_; uint8_t v___x_2662_; 
v___x_2645_ = lean_nat_add(v_lo_2631_, v_hi_2632_);
v___x_2646_ = lean_unsigned_to_nat(1u);
v_mid_2647_ = lean_nat_shiftr(v___x_2645_, v___x_2646_);
lean_dec(v___x_2645_);
v___x_2660_ = lean_array_fget_borrowed(v_as_2630_, v_mid_2647_);
v___x_2661_ = lean_array_fget_borrowed(v_as_2630_, v_lo_2631_);
v___x_2662_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___lam__0(v___x_2660_, v___x_2661_);
if (v___x_2662_ == 0)
{
v___y_2655_ = v_as_2630_;
goto v___jp_2654_;
}
else
{
lean_object* v___x_2663_; 
v___x_2663_ = lean_array_fswap(v_as_2630_, v_lo_2631_, v_mid_2647_);
v___y_2655_ = v___x_2663_;
goto v___jp_2654_;
}
v___jp_2648_:
{
lean_object* v___x_2650_; lean_object* v___x_2651_; uint8_t v___x_2652_; 
v___x_2650_ = lean_array_fget_borrowed(v___y_2649_, v_mid_2647_);
v___x_2651_ = lean_array_fget_borrowed(v___y_2649_, v_hi_2632_);
v___x_2652_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___lam__0(v___x_2650_, v___x_2651_);
if (v___x_2652_ == 0)
{
lean_dec(v_mid_2647_);
v___y_2634_ = v___y_2649_;
goto v___jp_2633_;
}
else
{
lean_object* v___x_2653_; 
v___x_2653_ = lean_array_fswap(v___y_2649_, v_mid_2647_, v_hi_2632_);
lean_dec(v_mid_2647_);
v___y_2634_ = v___x_2653_;
goto v___jp_2633_;
}
}
v___jp_2654_:
{
lean_object* v___x_2656_; lean_object* v___x_2657_; uint8_t v___x_2658_; 
v___x_2656_ = lean_array_fget_borrowed(v___y_2655_, v_hi_2632_);
v___x_2657_ = lean_array_fget_borrowed(v___y_2655_, v_lo_2631_);
v___x_2658_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___lam__0(v___x_2656_, v___x_2657_);
if (v___x_2658_ == 0)
{
v___y_2649_ = v___y_2655_;
goto v___jp_2648_;
}
else
{
lean_object* v___x_2659_; 
v___x_2659_ = lean_array_fswap(v___y_2655_, v_lo_2631_, v_hi_2632_);
v___y_2649_ = v___x_2659_;
goto v___jp_2648_;
}
}
}
v___jp_2633_:
{
lean_object* v_pivot_2635_; lean_object* v___x_2636_; lean_object* v_fst_2637_; lean_object* v_snd_2638_; uint8_t v___x_2639_; 
v_pivot_2635_ = lean_array_fget(v___y_2634_, v_hi_2632_);
lean_inc_n(v_lo_2631_, 2);
v___x_2636_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___redArg(v_hi_2632_, v_pivot_2635_, v___y_2634_, v_lo_2631_, v_lo_2631_);
lean_dec(v_pivot_2635_);
v_fst_2637_ = lean_ctor_get(v___x_2636_, 0);
lean_inc(v_fst_2637_);
v_snd_2638_ = lean_ctor_get(v___x_2636_, 1);
lean_inc(v_snd_2638_);
lean_dec_ref(v___x_2636_);
v___x_2639_ = lean_nat_dec_le(v_hi_2632_, v_fst_2637_);
if (v___x_2639_ == 0)
{
lean_object* v___x_2640_; lean_object* v___x_2641_; lean_object* v___x_2642_; 
v___x_2640_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg(v_n_2629_, v_snd_2638_, v_lo_2631_, v_fst_2637_);
v___x_2641_ = lean_unsigned_to_nat(1u);
v___x_2642_ = lean_nat_add(v_fst_2637_, v___x_2641_);
lean_dec(v_fst_2637_);
v_as_2630_ = v___x_2640_;
v_lo_2631_ = v___x_2642_;
goto _start;
}
else
{
lean_dec(v_fst_2637_);
lean_dec(v_lo_2631_);
return v_snd_2638_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg___boxed(lean_object* v_n_2664_, lean_object* v_as_2665_, lean_object* v_lo_2666_, lean_object* v_hi_2667_){
_start:
{
lean_object* v_res_2668_; 
v_res_2668_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg(v_n_2664_, v_as_2665_, v_lo_2666_, v_hi_2667_);
lean_dec(v_hi_2667_);
lean_dec(v_n_2664_);
return v_res_2668_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__0(void){
_start:
{
lean_object* v___x_2669_; 
v___x_2669_ = l_Array_instInhabited(lean_box(0));
return v___x_2669_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__1(void){
_start:
{
lean_object* v___x_2670_; lean_object* v___x_2671_; lean_object* v___x_2672_; 
v___x_2670_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__0, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__0);
v___x_2671_ = lean_box(0);
v___x_2672_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2672_, 0, v___x_2671_);
lean_ctor_set(v___x_2672_, 1, v___x_2670_);
return v___x_2672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs(lean_object* v_env_2675_){
_start:
{
lean_object* v___y_2677_; lean_object* v___y_2678_; lean_object* v___y_2679_; lean_object* v___y_2680_; lean_object* v___x_2684_; lean_object* v_toEnvExtension_2685_; lean_object* v_asyncMode_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v_tags_2689_; lean_object* v___y_2691_; lean_object* v_snd_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; uint8_t v___x_2704_; 
v___x_2684_ = lp_mathlib_Mathlib_CrossRef_tagExt;
v_toEnvExtension_2685_ = lean_ctor_get(v___x_2684_, 0);
v_asyncMode_2686_ = lean_ctor_get(v_toEnvExtension_2685_, 2);
v___x_2687_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__1, &lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__1);
v___x_2688_ = lean_box(0);
v_tags_2689_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_2687_, v___x_2684_, v_env_2675_, v_asyncMode_2686_, v___x_2688_);
v_snd_2700_ = lean_ctor_get(v_tags_2689_, 1);
lean_inc(v_snd_2700_);
v___x_2701_ = lean_unsigned_to_nat(0u);
v___x_2702_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs___closed__2));
v___x_2703_ = lean_array_get_size(v_snd_2700_);
v___x_2704_ = lean_nat_dec_lt(v___x_2701_, v___x_2703_);
if (v___x_2704_ == 0)
{
lean_dec(v_snd_2700_);
v___y_2691_ = v___x_2702_;
goto v___jp_2690_;
}
else
{
uint8_t v___x_2705_; 
v___x_2705_ = lean_nat_dec_le(v___x_2703_, v___x_2703_);
if (v___x_2705_ == 0)
{
if (v___x_2704_ == 0)
{
lean_dec(v_snd_2700_);
v___y_2691_ = v___x_2702_;
goto v___jp_2690_;
}
else
{
size_t v___x_2706_; size_t v___x_2707_; lean_object* v___x_2708_; 
v___x_2706_ = ((size_t)0ULL);
v___x_2707_ = lean_usize_of_nat(v___x_2703_);
v___x_2708_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__1(v_snd_2700_, v___x_2706_, v___x_2707_, v___x_2702_);
lean_dec(v_snd_2700_);
v___y_2691_ = v___x_2708_;
goto v___jp_2690_;
}
}
else
{
size_t v___x_2709_; size_t v___x_2710_; lean_object* v___x_2711_; 
v___x_2709_ = ((size_t)0ULL);
v___x_2710_ = lean_usize_of_nat(v___x_2703_);
v___x_2711_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__1(v_snd_2700_, v___x_2709_, v___x_2710_, v___x_2702_);
lean_dec(v_snd_2700_);
v___y_2691_ = v___x_2711_;
goto v___jp_2690_;
}
}
v___jp_2676_:
{
uint8_t v___x_2681_; 
v___x_2681_ = lean_nat_dec_le(v___y_2680_, v___y_2677_);
if (v___x_2681_ == 0)
{
lean_object* v___x_2682_; 
lean_dec(v___y_2677_);
lean_inc(v___y_2680_);
v___x_2682_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg(v___y_2678_, v___y_2679_, v___y_2680_, v___y_2680_);
lean_dec(v___y_2680_);
lean_dec(v___y_2678_);
return v___x_2682_;
}
else
{
lean_object* v___x_2683_; 
v___x_2683_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg(v___y_2678_, v___y_2679_, v___y_2680_, v___y_2677_);
lean_dec(v___y_2677_);
lean_dec(v___y_2678_);
return v___x_2683_;
}
}
v___jp_2690_:
{
lean_object* v_fst_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; uint8_t v___x_2696_; 
v_fst_2692_ = lean_ctor_get(v_tags_2689_, 0);
lean_inc(v_fst_2692_);
lean_dec(v_tags_2689_);
v___x_2693_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v___y_2691_, v_fst_2692_);
v___x_2694_ = lean_array_get_size(v___x_2693_);
v___x_2695_ = lean_unsigned_to_nat(0u);
v___x_2696_ = lean_nat_dec_eq(v___x_2694_, v___x_2695_);
if (v___x_2696_ == 0)
{
lean_object* v___x_2697_; lean_object* v___x_2698_; uint8_t v___x_2699_; 
v___x_2697_ = lean_unsigned_to_nat(1u);
v___x_2698_ = lean_nat_sub(v___x_2694_, v___x_2697_);
v___x_2699_ = lean_nat_dec_le(v___x_2695_, v___x_2698_);
if (v___x_2699_ == 0)
{
lean_inc(v___x_2698_);
v___y_2677_ = v___x_2698_;
v___y_2678_ = v___x_2694_;
v___y_2679_ = v___x_2693_;
v___y_2680_ = v___x_2698_;
goto v___jp_2676_;
}
else
{
v___y_2677_ = v___x_2698_;
v___y_2678_ = v___x_2694_;
v___y_2679_ = v___x_2693_;
v___y_2680_ = v___x_2695_;
goto v___jp_2676_;
}
}
else
{
return v___x_2693_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0(lean_object* v_n_2712_, lean_object* v_as_2713_, lean_object* v_lo_2714_, lean_object* v_hi_2715_, lean_object* v_w_2716_, lean_object* v_hlo_2717_, lean_object* v_hhi_2718_){
_start:
{
lean_object* v___x_2719_; 
v___x_2719_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___redArg(v_n_2712_, v_as_2713_, v_lo_2714_, v_hi_2715_);
return v___x_2719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0___boxed(lean_object* v_n_2720_, lean_object* v_as_2721_, lean_object* v_lo_2722_, lean_object* v_hi_2723_, lean_object* v_w_2724_, lean_object* v_hlo_2725_, lean_object* v_hhi_2726_){
_start:
{
lean_object* v_res_2727_; 
v_res_2727_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0(v_n_2720_, v_as_2721_, v_lo_2722_, v_hi_2723_, v_w_2724_, v_hlo_2725_, v_hhi_2726_);
lean_dec(v_hi_2723_);
lean_dec(v_n_2720_);
return v_res_2727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0(lean_object* v_n_2728_, lean_object* v_lo_2729_, lean_object* v_hi_2730_, lean_object* v_hhi_2731_, lean_object* v_pivot_2732_, lean_object* v_as_2733_, lean_object* v_i_2734_, lean_object* v_k_2735_, lean_object* v_ilo_2736_, lean_object* v_ik_2737_, lean_object* v_w_2738_){
_start:
{
lean_object* v___x_2739_; 
v___x_2739_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___redArg(v_hi_2730_, v_pivot_2732_, v_as_2733_, v_i_2734_, v_k_2735_);
return v___x_2739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0___boxed(lean_object* v_n_2740_, lean_object* v_lo_2741_, lean_object* v_hi_2742_, lean_object* v_hhi_2743_, lean_object* v_pivot_2744_, lean_object* v_as_2745_, lean_object* v_i_2746_, lean_object* v_k_2747_, lean_object* v_ilo_2748_, lean_object* v_ik_2749_, lean_object* v_w_2750_){
_start:
{
lean_object* v_res_2751_; 
v_res_2751_ = lp_mathlib___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs_spec__0_spec__0(v_n_2740_, v_lo_2741_, v_hi_2742_, v_hhi_2743_, v_pivot_2744_, v_as_2745_, v_i_2746_, v_k_2747_, v_ilo_2748_, v_ik_2749_, v_w_2750_);
lean_dec_ref(v_pivot_2744_);
lean_dec(v_hi_2742_);
lean_dec(v_lo_2741_);
lean_dec(v_n_2740_);
return v_res_2751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0_spec__0(lean_object* v_tag_2752_, lean_object* v_as_2753_, size_t v_i_2754_, size_t v_stop_2755_, lean_object* v_b_2756_){
_start:
{
lean_object* v___y_2758_; uint8_t v___x_2762_; 
v___x_2762_ = lean_usize_dec_eq(v_i_2754_, v_stop_2755_);
if (v___x_2762_ == 0)
{
lean_object* v___x_2763_; lean_object* v_declName_2764_; lean_object* v_tag_2765_; uint8_t v___x_2766_; 
v___x_2763_ = lean_array_uget_borrowed(v_as_2753_, v_i_2754_);
v_declName_2764_ = lean_ctor_get(v___x_2763_, 0);
v_tag_2765_ = lean_ctor_get(v___x_2763_, 2);
v___x_2766_ = lean_string_dec_eq(v_tag_2765_, v_tag_2752_);
if (v___x_2766_ == 0)
{
v___y_2758_ = v_b_2756_;
goto v___jp_2757_;
}
else
{
lean_object* v___x_2767_; 
lean_inc(v_declName_2764_);
v___x_2767_ = lean_array_push(v_b_2756_, v_declName_2764_);
v___y_2758_ = v___x_2767_;
goto v___jp_2757_;
}
}
else
{
return v_b_2756_;
}
v___jp_2757_:
{
size_t v___x_2759_; size_t v___x_2760_; 
v___x_2759_ = ((size_t)1ULL);
v___x_2760_ = lean_usize_add(v_i_2754_, v___x_2759_);
v_i_2754_ = v___x_2760_;
v_b_2756_ = v___y_2758_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0_spec__0___boxed(lean_object* v_tag_2768_, lean_object* v_as_2769_, lean_object* v_i_2770_, lean_object* v_stop_2771_, lean_object* v_b_2772_){
_start:
{
size_t v_i_boxed_2773_; size_t v_stop_boxed_2774_; lean_object* v_res_2775_; 
v_i_boxed_2773_ = lean_unbox_usize(v_i_2770_);
lean_dec(v_i_2770_);
v_stop_boxed_2774_ = lean_unbox_usize(v_stop_2771_);
lean_dec(v_stop_2771_);
v_res_2775_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0_spec__0(v_tag_2768_, v_as_2769_, v_i_boxed_2773_, v_stop_boxed_2774_, v_b_2772_);
lean_dec_ref(v_as_2769_);
lean_dec_ref(v_tag_2768_);
return v_res_2775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0(lean_object* v_tag_2778_, lean_object* v_as_2779_, lean_object* v_start_2780_, lean_object* v_stop_2781_){
_start:
{
lean_object* v___x_2782_; uint8_t v___x_2783_; 
v___x_2782_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0___closed__0));
v___x_2783_ = lean_nat_dec_lt(v_start_2780_, v_stop_2781_);
if (v___x_2783_ == 0)
{
return v___x_2782_;
}
else
{
lean_object* v___x_2784_; uint8_t v___x_2785_; 
v___x_2784_ = lean_array_get_size(v_as_2779_);
v___x_2785_ = lean_nat_dec_le(v_stop_2781_, v___x_2784_);
if (v___x_2785_ == 0)
{
uint8_t v___x_2786_; 
v___x_2786_ = lean_nat_dec_lt(v_start_2780_, v___x_2784_);
if (v___x_2786_ == 0)
{
return v___x_2782_;
}
else
{
size_t v___x_2787_; size_t v___x_2788_; lean_object* v___x_2789_; 
v___x_2787_ = lean_usize_of_nat(v_start_2780_);
v___x_2788_ = lean_usize_of_nat(v___x_2784_);
v___x_2789_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0_spec__0(v_tag_2778_, v_as_2779_, v___x_2787_, v___x_2788_, v___x_2782_);
return v___x_2789_;
}
}
else
{
size_t v___x_2790_; size_t v___x_2791_; lean_object* v___x_2792_; 
v___x_2790_ = lean_usize_of_nat(v_start_2780_);
v___x_2791_ = lean_usize_of_nat(v_stop_2781_);
v___x_2792_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0_spec__0(v_tag_2778_, v_as_2779_, v___x_2790_, v___x_2791_, v___x_2782_);
return v___x_2792_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0___boxed(lean_object* v_tag_2793_, lean_object* v_as_2794_, lean_object* v_start_2795_, lean_object* v_stop_2796_){
_start:
{
lean_object* v_res_2797_; 
v_res_2797_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0(v_tag_2793_, v_as_2794_, v_start_2795_, v_stop_2796_);
lean_dec(v_stop_2796_);
lean_dec(v_start_2795_);
lean_dec_ref(v_as_2794_);
lean_dec_ref(v_tag_2793_);
return v_res_2797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames(lean_object* v_env_2798_, lean_object* v_tag_2799_){
_start:
{
lean_object* v___x_2800_; lean_object* v___x_2801_; lean_object* v___x_2802_; lean_object* v___x_2803_; 
v___x_2800_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs(v_env_2798_);
v___x_2801_ = lean_unsigned_to_nat(0u);
v___x_2802_ = lean_array_get_size(v___x_2800_);
v___x_2803_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames_spec__0(v_tag_2799_, v___x_2800_, v___x_2801_, v___x_2802_);
lean_dec_ref(v___x_2800_);
return v___x_2803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames___boxed(lean_object* v_env_2804_, lean_object* v_tag_2805_){
_start:
{
lean_object* v_res_2806_; 
v_res_2806_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getCrossRefDeclNames(v_env_2804_, v_tag_2805_);
lean_dec_ref(v_tag_2805_);
return v_res_2806_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__5(lean_object* v_opts_2807_, lean_object* v_opt_2808_){
_start:
{
lean_object* v_name_2809_; lean_object* v_defValue_2810_; lean_object* v_map_2811_; lean_object* v___x_2812_; 
v_name_2809_ = lean_ctor_get(v_opt_2808_, 0);
v_defValue_2810_ = lean_ctor_get(v_opt_2808_, 1);
v_map_2811_ = lean_ctor_get(v_opts_2807_, 0);
v___x_2812_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2811_, v_name_2809_);
if (lean_obj_tag(v___x_2812_) == 0)
{
uint8_t v___x_2813_; 
v___x_2813_ = lean_unbox(v_defValue_2810_);
return v___x_2813_;
}
else
{
lean_object* v_val_2814_; 
v_val_2814_ = lean_ctor_get(v___x_2812_, 0);
lean_inc(v_val_2814_);
lean_dec_ref_known(v___x_2812_, 1);
if (lean_obj_tag(v_val_2814_) == 1)
{
uint8_t v_v_2815_; 
v_v_2815_ = lean_ctor_get_uint8(v_val_2814_, 0);
lean_dec_ref_known(v_val_2814_, 0);
return v_v_2815_;
}
else
{
uint8_t v___x_2816_; 
lean_dec(v_val_2814_);
v___x_2816_ = lean_unbox(v_defValue_2810_);
return v___x_2816_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__5___boxed(lean_object* v_opts_2817_, lean_object* v_opt_2818_){
_start:
{
uint8_t v_res_2819_; lean_object* v_r_2820_; 
v_res_2819_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__5(v_opts_2817_, v_opt_2818_);
lean_dec_ref(v_opt_2818_);
lean_dec_ref(v_opts_2817_);
v_r_2820_ = lean_box(v_res_2819_);
return v_r_2820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___redArg(lean_object* v_msgData_2821_, lean_object* v___y_2822_){
_start:
{
lean_object* v___x_2824_; lean_object* v_env_2825_; lean_object* v___x_2826_; lean_object* v_scopes_2827_; lean_object* v___x_2828_; lean_object* v___x_2829_; lean_object* v_opts_2830_; lean_object* v___x_2831_; lean_object* v___x_2832_; lean_object* v___x_2833_; lean_object* v___x_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; lean_object* v___x_2837_; 
v___x_2824_ = lean_st_ref_get(v___y_2822_);
v_env_2825_ = lean_ctor_get(v___x_2824_, 0);
lean_inc_ref(v_env_2825_);
lean_dec(v___x_2824_);
v___x_2826_ = lean_st_ref_get(v___y_2822_);
v_scopes_2827_ = lean_ctor_get(v___x_2826_, 2);
lean_inc(v_scopes_2827_);
lean_dec(v___x_2826_);
v___x_2828_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2829_ = l_List_head_x21___redArg(v___x_2828_, v_scopes_2827_);
lean_dec(v_scopes_2827_);
v_opts_2830_ = lean_ctor_get(v___x_2829_, 1);
lean_inc_ref(v_opts_2830_);
lean_dec(v___x_2829_);
v___x_2831_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__2);
v___x_2832_ = lean_unsigned_to_nat(32u);
v___x_2833_ = lean_mk_empty_array_with_capacity(v___x_2832_);
lean_dec_ref(v___x_2833_);
v___x_2834_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_throwError___at___00Lean_addDocStringCore___at___00Mathlib_CrossRef_addCrossRefDoc_spec__1_spec__1_spec__3___closed__5);
v___x_2835_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2835_, 0, v_env_2825_);
lean_ctor_set(v___x_2835_, 1, v___x_2831_);
lean_ctor_set(v___x_2835_, 2, v___x_2834_);
lean_ctor_set(v___x_2835_, 3, v_opts_2830_);
v___x_2836_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2836_, 0, v___x_2835_);
lean_ctor_set(v___x_2836_, 1, v_msgData_2821_);
v___x_2837_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2837_, 0, v___x_2836_);
return v___x_2837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_msgData_2838_, lean_object* v___y_2839_, lean_object* v___y_2840_){
_start:
{
lean_object* v_res_2841_; 
v_res_2841_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___redArg(v_msgData_2838_, v___y_2839_);
lean_dec(v___y_2839_);
return v_res_2841_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0(uint8_t v___y_2843_, uint8_t v_suppressElabErrors_2844_, lean_object* v_x_2845_){
_start:
{
if (lean_obj_tag(v_x_2845_) == 1)
{
lean_object* v_pre_2846_; 
v_pre_2846_ = lean_ctor_get(v_x_2845_, 0);
if (lean_obj_tag(v_pre_2846_) == 0)
{
lean_object* v_str_2847_; lean_object* v___x_2848_; uint8_t v___x_2849_; 
v_str_2847_ = lean_ctor_get(v_x_2845_, 1);
v___x_2848_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0___closed__0));
v___x_2849_ = lean_string_dec_eq(v_str_2847_, v___x_2848_);
if (v___x_2849_ == 0)
{
return v___y_2843_;
}
else
{
return v_suppressElabErrors_2844_;
}
}
else
{
return v___y_2843_;
}
}
else
{
return v___y_2843_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0___boxed(lean_object* v___y_2850_, lean_object* v_suppressElabErrors_2851_, lean_object* v_x_2852_){
_start:
{
uint8_t v___y_4244__boxed_2853_; uint8_t v_suppressElabErrors_boxed_2854_; uint8_t v_res_2855_; lean_object* v_r_2856_; 
v___y_4244__boxed_2853_ = lean_unbox(v___y_2850_);
v_suppressElabErrors_boxed_2854_ = lean_unbox(v_suppressElabErrors_2851_);
v_res_2855_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0(v___y_4244__boxed_2853_, v_suppressElabErrors_boxed_2854_, v_x_2852_);
lean_dec(v_x_2852_);
v_r_2856_ = lean_box(v_res_2855_);
return v_r_2856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2(lean_object* v_ref_2857_, lean_object* v_msgData_2858_, uint8_t v_severity_2859_, uint8_t v_isSilent_2860_, lean_object* v___y_2861_, lean_object* v___y_2862_){
_start:
{
lean_object* v___y_2865_; uint8_t v___y_2866_; lean_object* v___y_2867_; lean_object* v___y_2868_; lean_object* v___y_2869_; uint8_t v___y_2870_; lean_object* v___y_2871_; lean_object* v___y_2872_; uint8_t v___y_2929_; uint8_t v___y_2930_; uint8_t v___y_2931_; lean_object* v___y_2932_; lean_object* v___y_2933_; uint8_t v___y_2957_; uint8_t v___y_2958_; lean_object* v___y_2959_; uint8_t v___y_2960_; lean_object* v___y_2961_; uint8_t v___y_2965_; uint8_t v___y_2966_; uint8_t v___y_2967_; uint8_t v___x_2982_; uint8_t v___y_2984_; uint8_t v___y_2985_; uint8_t v___y_2986_; uint8_t v___y_2988_; uint8_t v___x_3000_; 
v___x_2982_ = 2;
v___x_3000_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2859_, v___x_2982_);
if (v___x_3000_ == 0)
{
v___y_2988_ = v___x_3000_;
goto v___jp_2987_;
}
else
{
uint8_t v___x_3001_; 
lean_inc_ref(v_msgData_2858_);
v___x_3001_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2858_);
v___y_2988_ = v___x_3001_;
goto v___jp_2987_;
}
v___jp_2864_:
{
lean_object* v___x_2873_; 
v___x_2873_ = l_Lean_Elab_Command_getScope___redArg(v___y_2872_);
if (lean_obj_tag(v___x_2873_) == 0)
{
lean_object* v_a_2874_; lean_object* v___x_2875_; 
v_a_2874_ = lean_ctor_get(v___x_2873_, 0);
lean_inc(v_a_2874_);
lean_dec_ref_known(v___x_2873_, 1);
v___x_2875_ = l_Lean_Elab_Command_getScope___redArg(v___y_2872_);
if (lean_obj_tag(v___x_2875_) == 0)
{
lean_object* v_a_2876_; lean_object* v___x_2878_; uint8_t v_isShared_2879_; uint8_t v_isSharedCheck_2911_; 
v_a_2876_ = lean_ctor_get(v___x_2875_, 0);
v_isSharedCheck_2911_ = !lean_is_exclusive(v___x_2875_);
if (v_isSharedCheck_2911_ == 0)
{
v___x_2878_ = v___x_2875_;
v_isShared_2879_ = v_isSharedCheck_2911_;
goto v_resetjp_2877_;
}
else
{
lean_inc(v_a_2876_);
lean_dec(v___x_2875_);
v___x_2878_ = lean_box(0);
v_isShared_2879_ = v_isSharedCheck_2911_;
goto v_resetjp_2877_;
}
v_resetjp_2877_:
{
lean_object* v___x_2880_; lean_object* v_currNamespace_2881_; lean_object* v_openDecls_2882_; lean_object* v_env_2883_; lean_object* v_messages_2884_; lean_object* v_scopes_2885_; lean_object* v_usedQuotCtxts_2886_; lean_object* v_nextMacroScope_2887_; lean_object* v_maxRecDepth_2888_; lean_object* v_ngen_2889_; lean_object* v_auxDeclNGen_2890_; lean_object* v_infoState_2891_; lean_object* v_traceState_2892_; lean_object* v_snapshotTasks_2893_; lean_object* v_prevLinterStates_2894_; lean_object* v___x_2896_; uint8_t v_isShared_2897_; uint8_t v_isSharedCheck_2910_; 
v___x_2880_ = lean_st_ref_take(v___y_2872_);
v_currNamespace_2881_ = lean_ctor_get(v_a_2874_, 2);
lean_inc(v_currNamespace_2881_);
lean_dec(v_a_2874_);
v_openDecls_2882_ = lean_ctor_get(v_a_2876_, 3);
lean_inc(v_openDecls_2882_);
lean_dec(v_a_2876_);
v_env_2883_ = lean_ctor_get(v___x_2880_, 0);
v_messages_2884_ = lean_ctor_get(v___x_2880_, 1);
v_scopes_2885_ = lean_ctor_get(v___x_2880_, 2);
v_usedQuotCtxts_2886_ = lean_ctor_get(v___x_2880_, 3);
v_nextMacroScope_2887_ = lean_ctor_get(v___x_2880_, 4);
v_maxRecDepth_2888_ = lean_ctor_get(v___x_2880_, 5);
v_ngen_2889_ = lean_ctor_get(v___x_2880_, 6);
v_auxDeclNGen_2890_ = lean_ctor_get(v___x_2880_, 7);
v_infoState_2891_ = lean_ctor_get(v___x_2880_, 8);
v_traceState_2892_ = lean_ctor_get(v___x_2880_, 9);
v_snapshotTasks_2893_ = lean_ctor_get(v___x_2880_, 10);
v_prevLinterStates_2894_ = lean_ctor_get(v___x_2880_, 11);
v_isSharedCheck_2910_ = !lean_is_exclusive(v___x_2880_);
if (v_isSharedCheck_2910_ == 0)
{
v___x_2896_ = v___x_2880_;
v_isShared_2897_ = v_isSharedCheck_2910_;
goto v_resetjp_2895_;
}
else
{
lean_inc(v_prevLinterStates_2894_);
lean_inc(v_snapshotTasks_2893_);
lean_inc(v_traceState_2892_);
lean_inc(v_infoState_2891_);
lean_inc(v_auxDeclNGen_2890_);
lean_inc(v_ngen_2889_);
lean_inc(v_maxRecDepth_2888_);
lean_inc(v_nextMacroScope_2887_);
lean_inc(v_usedQuotCtxts_2886_);
lean_inc(v_scopes_2885_);
lean_inc(v_messages_2884_);
lean_inc(v_env_2883_);
lean_dec(v___x_2880_);
v___x_2896_ = lean_box(0);
v_isShared_2897_ = v_isSharedCheck_2910_;
goto v_resetjp_2895_;
}
v_resetjp_2895_:
{
lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; lean_object* v___x_2903_; 
v___x_2898_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2898_, 0, v_currNamespace_2881_);
lean_ctor_set(v___x_2898_, 1, v_openDecls_2882_);
v___x_2899_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2899_, 0, v___x_2898_);
lean_ctor_set(v___x_2899_, 1, v___y_2867_);
lean_inc_ref(v___y_2868_);
lean_inc_ref(v___y_2865_);
v___x_2900_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_2900_, 0, v___y_2865_);
lean_ctor_set(v___x_2900_, 1, v___y_2869_);
lean_ctor_set(v___x_2900_, 2, v___y_2871_);
lean_ctor_set(v___x_2900_, 3, v___y_2868_);
lean_ctor_set(v___x_2900_, 4, v___x_2899_);
lean_ctor_set_uint8(v___x_2900_, sizeof(void*)*5, v___y_2870_);
lean_ctor_set_uint8(v___x_2900_, sizeof(void*)*5 + 1, v___y_2866_);
lean_ctor_set_uint8(v___x_2900_, sizeof(void*)*5 + 2, v_isSilent_2860_);
v___x_2901_ = l_Lean_MessageLog_add(v___x_2900_, v_messages_2884_);
if (v_isShared_2897_ == 0)
{
lean_ctor_set(v___x_2896_, 1, v___x_2901_);
v___x_2903_ = v___x_2896_;
goto v_reusejp_2902_;
}
else
{
lean_object* v_reuseFailAlloc_2909_; 
v_reuseFailAlloc_2909_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_2909_, 0, v_env_2883_);
lean_ctor_set(v_reuseFailAlloc_2909_, 1, v___x_2901_);
lean_ctor_set(v_reuseFailAlloc_2909_, 2, v_scopes_2885_);
lean_ctor_set(v_reuseFailAlloc_2909_, 3, v_usedQuotCtxts_2886_);
lean_ctor_set(v_reuseFailAlloc_2909_, 4, v_nextMacroScope_2887_);
lean_ctor_set(v_reuseFailAlloc_2909_, 5, v_maxRecDepth_2888_);
lean_ctor_set(v_reuseFailAlloc_2909_, 6, v_ngen_2889_);
lean_ctor_set(v_reuseFailAlloc_2909_, 7, v_auxDeclNGen_2890_);
lean_ctor_set(v_reuseFailAlloc_2909_, 8, v_infoState_2891_);
lean_ctor_set(v_reuseFailAlloc_2909_, 9, v_traceState_2892_);
lean_ctor_set(v_reuseFailAlloc_2909_, 10, v_snapshotTasks_2893_);
lean_ctor_set(v_reuseFailAlloc_2909_, 11, v_prevLinterStates_2894_);
v___x_2903_ = v_reuseFailAlloc_2909_;
goto v_reusejp_2902_;
}
v_reusejp_2902_:
{
lean_object* v___x_2904_; lean_object* v___x_2905_; lean_object* v___x_2907_; 
v___x_2904_ = lean_st_ref_set(v___y_2872_, v___x_2903_);
v___x_2905_ = lean_box(0);
if (v_isShared_2879_ == 0)
{
lean_ctor_set(v___x_2878_, 0, v___x_2905_);
v___x_2907_ = v___x_2878_;
goto v_reusejp_2906_;
}
else
{
lean_object* v_reuseFailAlloc_2908_; 
v_reuseFailAlloc_2908_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2908_, 0, v___x_2905_);
v___x_2907_ = v_reuseFailAlloc_2908_;
goto v_reusejp_2906_;
}
v_reusejp_2906_:
{
return v___x_2907_;
}
}
}
}
}
else
{
lean_object* v_a_2912_; lean_object* v___x_2914_; uint8_t v_isShared_2915_; uint8_t v_isSharedCheck_2919_; 
lean_dec(v_a_2874_);
lean_dec(v___y_2871_);
lean_dec_ref(v___y_2869_);
lean_dec_ref(v___y_2867_);
v_a_2912_ = lean_ctor_get(v___x_2875_, 0);
v_isSharedCheck_2919_ = !lean_is_exclusive(v___x_2875_);
if (v_isSharedCheck_2919_ == 0)
{
v___x_2914_ = v___x_2875_;
v_isShared_2915_ = v_isSharedCheck_2919_;
goto v_resetjp_2913_;
}
else
{
lean_inc(v_a_2912_);
lean_dec(v___x_2875_);
v___x_2914_ = lean_box(0);
v_isShared_2915_ = v_isSharedCheck_2919_;
goto v_resetjp_2913_;
}
v_resetjp_2913_:
{
lean_object* v___x_2917_; 
if (v_isShared_2915_ == 0)
{
v___x_2917_ = v___x_2914_;
goto v_reusejp_2916_;
}
else
{
lean_object* v_reuseFailAlloc_2918_; 
v_reuseFailAlloc_2918_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2918_, 0, v_a_2912_);
v___x_2917_ = v_reuseFailAlloc_2918_;
goto v_reusejp_2916_;
}
v_reusejp_2916_:
{
return v___x_2917_;
}
}
}
}
else
{
lean_object* v_a_2920_; lean_object* v___x_2922_; uint8_t v_isShared_2923_; uint8_t v_isSharedCheck_2927_; 
lean_dec(v___y_2871_);
lean_dec_ref(v___y_2869_);
lean_dec_ref(v___y_2867_);
v_a_2920_ = lean_ctor_get(v___x_2873_, 0);
v_isSharedCheck_2927_ = !lean_is_exclusive(v___x_2873_);
if (v_isSharedCheck_2927_ == 0)
{
v___x_2922_ = v___x_2873_;
v_isShared_2923_ = v_isSharedCheck_2927_;
goto v_resetjp_2921_;
}
else
{
lean_inc(v_a_2920_);
lean_dec(v___x_2873_);
v___x_2922_ = lean_box(0);
v_isShared_2923_ = v_isSharedCheck_2927_;
goto v_resetjp_2921_;
}
v_resetjp_2921_:
{
lean_object* v___x_2925_; 
if (v_isShared_2923_ == 0)
{
v___x_2925_ = v___x_2922_;
goto v_reusejp_2924_;
}
else
{
lean_object* v_reuseFailAlloc_2926_; 
v_reuseFailAlloc_2926_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2926_, 0, v_a_2920_);
v___x_2925_ = v_reuseFailAlloc_2926_;
goto v_reusejp_2924_;
}
v_reusejp_2924_:
{
return v___x_2925_;
}
}
}
}
v___jp_2928_:
{
lean_object* v_fileName_2934_; lean_object* v_fileMap_2935_; uint8_t v_suppressElabErrors_2936_; lean_object* v___x_2937_; lean_object* v___x_2938_; lean_object* v_a_2939_; lean_object* v___x_2941_; uint8_t v_isShared_2942_; uint8_t v_isSharedCheck_2955_; 
v_fileName_2934_ = lean_ctor_get(v___y_2861_, 0);
v_fileMap_2935_ = lean_ctor_get(v___y_2861_, 1);
v_suppressElabErrors_2936_ = lean_ctor_get_uint8(v___y_2861_, sizeof(void*)*10);
v___x_2937_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2858_);
v___x_2938_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___redArg(v___x_2937_, v___y_2862_);
v_a_2939_ = lean_ctor_get(v___x_2938_, 0);
v_isSharedCheck_2955_ = !lean_is_exclusive(v___x_2938_);
if (v_isSharedCheck_2955_ == 0)
{
v___x_2941_ = v___x_2938_;
v_isShared_2942_ = v_isSharedCheck_2955_;
goto v_resetjp_2940_;
}
else
{
lean_inc(v_a_2939_);
lean_dec(v___x_2938_);
v___x_2941_ = lean_box(0);
v_isShared_2942_ = v_isSharedCheck_2955_;
goto v_resetjp_2940_;
}
v_resetjp_2940_:
{
lean_object* v___x_2943_; lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2946_; 
lean_inc_ref_n(v_fileMap_2935_, 2);
v___x_2943_ = l_Lean_FileMap_toPosition(v_fileMap_2935_, v___y_2932_);
lean_dec(v___y_2932_);
v___x_2944_ = l_Lean_FileMap_toPosition(v_fileMap_2935_, v___y_2933_);
lean_dec(v___y_2933_);
v___x_2945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2945_, 0, v___x_2944_);
v___x_2946_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
if (v_suppressElabErrors_2936_ == 0)
{
lean_del_object(v___x_2941_);
v___y_2865_ = v_fileName_2934_;
v___y_2866_ = v___y_2930_;
v___y_2867_ = v_a_2939_;
v___y_2868_ = v___x_2946_;
v___y_2869_ = v___x_2943_;
v___y_2870_ = v___y_2931_;
v___y_2871_ = v___x_2945_;
v___y_2872_ = v___y_2862_;
goto v___jp_2864_;
}
else
{
lean_object* v___x_2947_; lean_object* v___x_2948_; lean_object* v___f_2949_; uint8_t v___x_2950_; 
v___x_2947_ = lean_box(v___y_2929_);
v___x_2948_ = lean_box(v_suppressElabErrors_2936_);
v___f_2949_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2949_, 0, v___x_2947_);
lean_closure_set(v___f_2949_, 1, v___x_2948_);
lean_inc(v_a_2939_);
v___x_2950_ = l_Lean_MessageData_hasTag(v___f_2949_, v_a_2939_);
if (v___x_2950_ == 0)
{
lean_object* v___x_2951_; lean_object* v___x_2953_; 
lean_dec_ref_known(v___x_2945_, 1);
lean_dec_ref(v___x_2943_);
lean_dec(v_a_2939_);
v___x_2951_ = lean_box(0);
if (v_isShared_2942_ == 0)
{
lean_ctor_set(v___x_2941_, 0, v___x_2951_);
v___x_2953_ = v___x_2941_;
goto v_reusejp_2952_;
}
else
{
lean_object* v_reuseFailAlloc_2954_; 
v_reuseFailAlloc_2954_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2954_, 0, v___x_2951_);
v___x_2953_ = v_reuseFailAlloc_2954_;
goto v_reusejp_2952_;
}
v_reusejp_2952_:
{
return v___x_2953_;
}
}
else
{
lean_del_object(v___x_2941_);
v___y_2865_ = v_fileName_2934_;
v___y_2866_ = v___y_2930_;
v___y_2867_ = v_a_2939_;
v___y_2868_ = v___x_2946_;
v___y_2869_ = v___x_2943_;
v___y_2870_ = v___y_2931_;
v___y_2871_ = v___x_2945_;
v___y_2872_ = v___y_2862_;
goto v___jp_2864_;
}
}
}
}
v___jp_2956_:
{
lean_object* v___x_2962_; 
v___x_2962_ = l_Lean_Syntax_getTailPos_x3f(v___y_2959_, v___y_2960_);
lean_dec(v___y_2959_);
if (lean_obj_tag(v___x_2962_) == 0)
{
lean_inc(v___y_2961_);
v___y_2929_ = v___y_2957_;
v___y_2930_ = v___y_2958_;
v___y_2931_ = v___y_2960_;
v___y_2932_ = v___y_2961_;
v___y_2933_ = v___y_2961_;
goto v___jp_2928_;
}
else
{
lean_object* v_val_2963_; 
v_val_2963_ = lean_ctor_get(v___x_2962_, 0);
lean_inc(v_val_2963_);
lean_dec_ref_known(v___x_2962_, 1);
v___y_2929_ = v___y_2957_;
v___y_2930_ = v___y_2958_;
v___y_2931_ = v___y_2960_;
v___y_2932_ = v___y_2961_;
v___y_2933_ = v_val_2963_;
goto v___jp_2928_;
}
}
v___jp_2964_:
{
lean_object* v___x_2968_; 
v___x_2968_ = l_Lean_Elab_Command_getRef___redArg(v___y_2861_);
if (lean_obj_tag(v___x_2968_) == 0)
{
lean_object* v_a_2969_; lean_object* v_ref_2970_; lean_object* v___x_2971_; 
v_a_2969_ = lean_ctor_get(v___x_2968_, 0);
lean_inc(v_a_2969_);
lean_dec_ref_known(v___x_2968_, 1);
v_ref_2970_ = l_Lean_replaceRef(v_ref_2857_, v_a_2969_);
lean_dec(v_a_2969_);
v___x_2971_ = l_Lean_Syntax_getPos_x3f(v_ref_2970_, v___y_2966_);
if (lean_obj_tag(v___x_2971_) == 0)
{
lean_object* v___x_2972_; 
v___x_2972_ = lean_unsigned_to_nat(0u);
v___y_2957_ = v___y_2965_;
v___y_2958_ = v___y_2967_;
v___y_2959_ = v_ref_2970_;
v___y_2960_ = v___y_2966_;
v___y_2961_ = v___x_2972_;
goto v___jp_2956_;
}
else
{
lean_object* v_val_2973_; 
v_val_2973_ = lean_ctor_get(v___x_2971_, 0);
lean_inc(v_val_2973_);
lean_dec_ref_known(v___x_2971_, 1);
v___y_2957_ = v___y_2965_;
v___y_2958_ = v___y_2967_;
v___y_2959_ = v_ref_2970_;
v___y_2960_ = v___y_2966_;
v___y_2961_ = v_val_2973_;
goto v___jp_2956_;
}
}
else
{
lean_object* v_a_2974_; lean_object* v___x_2976_; uint8_t v_isShared_2977_; uint8_t v_isSharedCheck_2981_; 
lean_dec_ref(v_msgData_2858_);
v_a_2974_ = lean_ctor_get(v___x_2968_, 0);
v_isSharedCheck_2981_ = !lean_is_exclusive(v___x_2968_);
if (v_isSharedCheck_2981_ == 0)
{
v___x_2976_ = v___x_2968_;
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
else
{
lean_inc(v_a_2974_);
lean_dec(v___x_2968_);
v___x_2976_ = lean_box(0);
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
v_resetjp_2975_:
{
lean_object* v___x_2979_; 
if (v_isShared_2977_ == 0)
{
v___x_2979_ = v___x_2976_;
goto v_reusejp_2978_;
}
else
{
lean_object* v_reuseFailAlloc_2980_; 
v_reuseFailAlloc_2980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2980_, 0, v_a_2974_);
v___x_2979_ = v_reuseFailAlloc_2980_;
goto v_reusejp_2978_;
}
v_reusejp_2978_:
{
return v___x_2979_;
}
}
}
}
v___jp_2983_:
{
if (v___y_2986_ == 0)
{
v___y_2965_ = v___y_2984_;
v___y_2966_ = v___y_2985_;
v___y_2967_ = v_severity_2859_;
goto v___jp_2964_;
}
else
{
v___y_2965_ = v___y_2984_;
v___y_2966_ = v___y_2985_;
v___y_2967_ = v___x_2982_;
goto v___jp_2964_;
}
}
v___jp_2987_:
{
if (v___y_2988_ == 0)
{
lean_object* v___x_2989_; lean_object* v_scopes_2990_; lean_object* v___x_2991_; lean_object* v___x_2992_; lean_object* v_opts_2993_; uint8_t v___x_2994_; uint8_t v___x_2995_; 
v___x_2989_ = lean_st_ref_get(v___y_2862_);
v_scopes_2990_ = lean_ctor_get(v___x_2989_, 2);
lean_inc(v_scopes_2990_);
lean_dec(v___x_2989_);
v___x_2991_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2992_ = l_List_head_x21___redArg(v___x_2991_, v_scopes_2990_);
lean_dec(v_scopes_2990_);
v_opts_2993_ = lean_ctor_get(v___x_2992_, 1);
lean_inc_ref(v_opts_2993_);
lean_dec(v___x_2992_);
v___x_2994_ = 1;
v___x_2995_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2859_, v___x_2994_);
if (v___x_2995_ == 0)
{
lean_dec_ref(v_opts_2993_);
v___y_2984_ = v___y_2988_;
v___y_2985_ = v___y_2988_;
v___y_2986_ = v___x_2995_;
goto v___jp_2983_;
}
else
{
lean_object* v___x_2996_; uint8_t v___x_2997_; 
v___x_2996_ = l_Lean_warningAsError;
v___x_2997_ = lp_mathlib_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__5(v_opts_2993_, v___x_2996_);
lean_dec_ref(v_opts_2993_);
v___y_2984_ = v___y_2988_;
v___y_2985_ = v___y_2988_;
v___y_2986_ = v___x_2997_;
goto v___jp_2983_;
}
}
else
{
lean_object* v___x_2998_; lean_object* v___x_2999_; 
lean_dec_ref(v_msgData_2858_);
v___x_2998_ = lean_box(0);
v___x_2999_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2999_, 0, v___x_2998_);
return v___x_2999_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2___boxed(lean_object* v_ref_3002_, lean_object* v_msgData_3003_, lean_object* v_severity_3004_, lean_object* v_isSilent_3005_, lean_object* v___y_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_){
_start:
{
uint8_t v_severity_boxed_3009_; uint8_t v_isSilent_boxed_3010_; lean_object* v_res_3011_; 
v_severity_boxed_3009_ = lean_unbox(v_severity_3004_);
v_isSilent_boxed_3010_ = lean_unbox(v_isSilent_3005_);
v_res_3011_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2(v_ref_3002_, v_msgData_3003_, v_severity_boxed_3009_, v_isSilent_boxed_3010_, v___y_3006_, v___y_3007_);
lean_dec(v___y_3007_);
lean_dec_ref(v___y_3006_);
lean_dec(v_ref_3002_);
return v_res_3011_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1(lean_object* v_msgData_3012_, uint8_t v_severity_3013_, uint8_t v_isSilent_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_){
_start:
{
lean_object* v___x_3018_; 
v___x_3018_ = l_Lean_Elab_Command_getRef___redArg(v___y_3015_);
if (lean_obj_tag(v___x_3018_) == 0)
{
lean_object* v_a_3019_; lean_object* v___x_3020_; 
v_a_3019_ = lean_ctor_get(v___x_3018_, 0);
lean_inc(v_a_3019_);
lean_dec_ref_known(v___x_3018_, 1);
v___x_3020_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2(v_a_3019_, v_msgData_3012_, v_severity_3013_, v_isSilent_3014_, v___y_3015_, v___y_3016_);
lean_dec(v_a_3019_);
return v___x_3020_;
}
else
{
lean_object* v_a_3021_; lean_object* v___x_3023_; uint8_t v_isShared_3024_; uint8_t v_isSharedCheck_3028_; 
lean_dec_ref(v_msgData_3012_);
v_a_3021_ = lean_ctor_get(v___x_3018_, 0);
v_isSharedCheck_3028_ = !lean_is_exclusive(v___x_3018_);
if (v_isSharedCheck_3028_ == 0)
{
v___x_3023_ = v___x_3018_;
v_isShared_3024_ = v_isSharedCheck_3028_;
goto v_resetjp_3022_;
}
else
{
lean_inc(v_a_3021_);
lean_dec(v___x_3018_);
v___x_3023_ = lean_box(0);
v_isShared_3024_ = v_isSharedCheck_3028_;
goto v_resetjp_3022_;
}
v_resetjp_3022_:
{
lean_object* v___x_3026_; 
if (v_isShared_3024_ == 0)
{
v___x_3026_ = v___x_3023_;
goto v_reusejp_3025_;
}
else
{
lean_object* v_reuseFailAlloc_3027_; 
v_reuseFailAlloc_3027_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3027_, 0, v_a_3021_);
v___x_3026_ = v_reuseFailAlloc_3027_;
goto v_reusejp_3025_;
}
v_reusejp_3025_:
{
return v___x_3026_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1___boxed(lean_object* v_msgData_3029_, lean_object* v_severity_3030_, lean_object* v_isSilent_3031_, lean_object* v___y_3032_, lean_object* v___y_3033_, lean_object* v___y_3034_){
_start:
{
uint8_t v_severity_boxed_3035_; uint8_t v_isSilent_boxed_3036_; lean_object* v_res_3037_; 
v_severity_boxed_3035_ = lean_unbox(v_severity_3030_);
v_isSilent_boxed_3036_ = lean_unbox(v_isSilent_3031_);
v_res_3037_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1(v_msgData_3029_, v_severity_boxed_3035_, v_isSilent_boxed_3036_, v___y_3032_, v___y_3033_);
lean_dec(v___y_3033_);
lean_dec_ref(v___y_3032_);
return v_res_3037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1(lean_object* v_msgData_3038_, lean_object* v___y_3039_, lean_object* v___y_3040_){
_start:
{
uint8_t v___x_3042_; uint8_t v___x_3043_; lean_object* v___x_3044_; 
v___x_3042_ = 0;
v___x_3043_ = 0;
v___x_3044_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1(v_msgData_3038_, v___x_3042_, v___x_3043_, v___y_3039_, v___y_3040_);
return v___x_3044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1___boxed(lean_object* v_msgData_3045_, lean_object* v___y_3046_, lean_object* v___y_3047_, lean_object* v___y_3048_){
_start:
{
lean_object* v_res_3049_; 
v_res_3049_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1(v_msgData_3045_, v___y_3046_, v___y_3047_);
lean_dec(v___y_3047_);
lean_dec_ref(v___y_3046_);
return v_res_3049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_CrossRef_traceCrossRefs_spec__2(lean_object* v_db_3050_, lean_object* v_as_3051_, size_t v_i_3052_, size_t v_stop_3053_, lean_object* v_b_3054_){
_start:
{
lean_object* v___y_3056_; uint8_t v___x_3060_; 
v___x_3060_ = lean_usize_dec_eq(v_i_3052_, v_stop_3053_);
if (v___x_3060_ == 0)
{
lean_object* v___x_3061_; lean_object* v_database_3062_; uint8_t v___x_3063_; 
v___x_3061_ = lean_array_uget_borrowed(v_as_3051_, v_i_3052_);
v_database_3062_ = lean_ctor_get(v___x_3061_, 1);
v___x_3063_ = lp_mathlib_Mathlib_CrossRef_instBEqDatabase_beq(v_database_3062_, v_db_3050_);
if (v___x_3063_ == 0)
{
v___y_3056_ = v_b_3054_;
goto v___jp_3055_;
}
else
{
lean_object* v___x_3064_; 
lean_inc(v___x_3061_);
v___x_3064_ = lean_array_push(v_b_3054_, v___x_3061_);
v___y_3056_ = v___x_3064_;
goto v___jp_3055_;
}
}
else
{
return v_b_3054_;
}
v___jp_3055_:
{
size_t v___x_3057_; size_t v___x_3058_; 
v___x_3057_ = ((size_t)1ULL);
v___x_3058_ = lean_usize_add(v_i_3052_, v___x_3057_);
v_i_3052_ = v___x_3058_;
v_b_3054_ = v___y_3056_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_CrossRef_traceCrossRefs_spec__2___boxed(lean_object* v_db_3065_, lean_object* v_as_3066_, lean_object* v_i_3067_, lean_object* v_stop_3068_, lean_object* v_b_3069_){
_start:
{
size_t v_i_boxed_3070_; size_t v_stop_boxed_3071_; lean_object* v_res_3072_; 
v_i_boxed_3070_ = lean_unbox_usize(v_i_3067_);
lean_dec(v_i_3067_);
v_stop_boxed_3071_ = lean_unbox_usize(v_stop_3068_);
lean_dec(v_stop_3068_);
v_res_3072_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_CrossRef_traceCrossRefs_spec__2(v_db_3065_, v_as_3066_, v_i_boxed_3070_, v_stop_boxed_3071_, v_b_3069_);
lean_dec_ref(v_as_3066_);
lean_dec(v_db_3065_);
return v_res_3072_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_3075_; lean_object* v___x_3076_; 
v___x_3075_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__0));
v___x_3076_ = l_Lean_MessageData_ofFormat(v___x_3075_);
return v___x_3076_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_3077_; lean_object* v___x_3078_; 
v___x_3077_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__0));
v___x_3078_ = l_Lean_stringToMessageData(v___x_3077_);
return v___x_3078_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_3079_; lean_object* v___x_3080_; 
v___x_3079_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__1));
v___x_3080_ = l_Lean_stringToMessageData(v___x_3079_);
return v___x_3080_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_3081_; lean_object* v___x_3082_; 
v___x_3081_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__2));
v___x_3082_ = l_Lean_stringToMessageData(v___x_3081_);
return v___x_3082_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__6(void){
_start:
{
lean_object* v___x_3084_; lean_object* v___x_3085_; 
v___x_3084_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__5));
v___x_3085_ = l_Lean_stringToMessageData(v___x_3084_);
return v___x_3085_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__8(void){
_start:
{
lean_object* v___x_3087_; lean_object* v___x_3088_; 
v___x_3087_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__7));
v___x_3088_ = l_Lean_stringToMessageData(v___x_3087_);
return v___x_3088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg(lean_object* v_db_3089_, lean_object* v___x_3090_, uint8_t v_verbose_3091_, lean_object* v___x_3092_, lean_object* v_as_3093_, size_t v_sz_3094_, size_t v_i_3095_, lean_object* v_b_3096_){
_start:
{
lean_object* v_a_3099_; uint8_t v___x_3103_; 
v___x_3103_ = lean_usize_dec_lt(v_i_3095_, v_sz_3094_);
if (v___x_3103_ == 0)
{
lean_object* v___x_3104_; 
lean_dec_ref(v___x_3092_);
v___x_3104_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3104_, 0, v_b_3096_);
return v___x_3104_;
}
else
{
lean_object* v_a_3105_; lean_object* v_comment_3106_; lean_object* v___x_3107_; uint8_t v___x_3108_; lean_object* v___x_3109_; lean_object* v___y_3111_; lean_object* v___y_3112_; lean_object* v_fst_3119_; lean_object* v_snd_3120_; lean_object* v___x_3151_; uint8_t v___x_3152_; 
v_a_3105_ = lean_array_uget_borrowed(v_as_3093_, v_i_3095_);
v_comment_3106_ = lean_ctor_get(v_a_3105_, 3);
v___x_3107_ = lean_unsigned_to_nat(0u);
v___x_3108_ = lean_nat_dec_eq(v___x_3090_, v___x_3107_);
v___x_3109_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___x_3151_ = lean_string_utf8_byte_size(v_comment_3106_);
v___x_3152_ = lean_nat_dec_eq(v___x_3151_, v___x_3107_);
if (v___x_3152_ == 0)
{
lean_object* v___x_3153_; lean_object* v___x_3154_; 
v___x_3153_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__5));
v___x_3154_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_addCrossRefDoc___closed__3));
v_fst_3119_ = v___x_3153_;
v_snd_3120_ = v___x_3154_;
goto v___jp_3118_;
}
else
{
v_fst_3119_ = v___x_3109_;
v_snd_3120_ = v___x_3109_;
goto v___jp_3118_;
}
v___jp_3110_:
{
lean_object* v___x_3113_; lean_object* v___x_3114_; lean_object* v___x_3115_; lean_object* v___x_3116_; lean_object* v___x_3117_; 
v___x_3113_ = l_Lean_ConstantInfo_type(v___y_3112_);
lean_dec_ref(v___y_3112_);
v___x_3114_ = l_Lean_MessageData_ofExpr(v___x_3113_);
v___x_3115_ = lean_array_push(v___y_3111_, v___x_3114_);
v___x_3116_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__1);
v___x_3117_ = lean_array_push(v___x_3115_, v___x_3116_);
v_a_3099_ = v___x_3117_;
goto v___jp_3098_;
}
v___jp_3118_:
{
lean_object* v_declName_3121_; lean_object* v_tag_3122_; lean_object* v_comment_3123_; lean_object* v___x_3124_; lean_object* v___x_3125_; lean_object* v___x_3126_; lean_object* v___x_3127_; lean_object* v___x_3128_; lean_object* v___x_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; lean_object* v___x_3132_; lean_object* v___x_3133_; lean_object* v___x_3134_; lean_object* v___x_3135_; lean_object* v___x_3136_; lean_object* v___x_3137_; lean_object* v___x_3138_; lean_object* v___x_3139_; lean_object* v___x_3140_; lean_object* v___x_3141_; lean_object* v___x_3142_; lean_object* v___x_3143_; lean_object* v___x_3144_; lean_object* v___x_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; 
v_declName_3121_ = lean_ctor_get(v_a_3105_, 0);
v_tag_3122_ = lean_ctor_get(v_a_3105_, 2);
v_comment_3123_ = lean_ctor_get(v_a_3105_, 3);
lean_inc_ref(v_fst_3119_);
v___x_3124_ = lean_string_append(v_fst_3119_, v_comment_3123_);
v___x_3125_ = lean_string_append(v___x_3124_, v_snd_3120_);
v___x_3126_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__2, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__2_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__2);
v___x_3127_ = lp_mathlib_Mathlib_CrossRef_Database_label(v_db_3089_);
v___x_3128_ = l_Lean_stringToMessageData(v___x_3127_);
v___x_3129_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3129_, 0, v___x_3126_);
lean_ctor_set(v___x_3129_, 1, v___x_3128_);
v___x_3130_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__3);
v___x_3131_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3131_, 0, v___x_3129_);
lean_ctor_set(v___x_3131_, 1, v___x_3130_);
lean_inc_ref_n(v_tag_3122_, 2);
v___x_3132_ = l_Lean_stringToMessageData(v_tag_3122_);
v___x_3133_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3133_, 0, v___x_3131_);
lean_ctor_set(v___x_3133_, 1, v___x_3132_);
v___x_3134_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__4);
v___x_3135_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3135_, 0, v___x_3133_);
lean_ctor_set(v___x_3135_, 1, v___x_3134_);
v___x_3136_ = lp_mathlib_Mathlib_CrossRef_Database_url(v_db_3089_, v_tag_3122_);
v___x_3137_ = l_Lean_stringToMessageData(v___x_3136_);
v___x_3138_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3138_, 0, v___x_3135_);
lean_ctor_set(v___x_3138_, 1, v___x_3137_);
v___x_3139_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__6, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__6_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__6);
v___x_3140_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3140_, 0, v___x_3138_);
lean_ctor_set(v___x_3140_, 1, v___x_3139_);
lean_inc(v_declName_3121_);
v___x_3141_ = l_Lean_MessageData_ofConstName(v_declName_3121_, v___x_3108_);
v___x_3142_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3142_, 0, v___x_3140_);
lean_ctor_set(v___x_3142_, 1, v___x_3141_);
v___x_3143_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__8, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__8_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___closed__8);
v___x_3144_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3144_, 0, v___x_3142_);
lean_ctor_set(v___x_3144_, 1, v___x_3143_);
v___x_3145_ = l_Lean_stringToMessageData(v___x_3125_);
v___x_3146_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3146_, 0, v___x_3144_);
lean_ctor_set(v___x_3146_, 1, v___x_3145_);
v___x_3147_ = lean_array_push(v_b_3096_, v___x_3146_);
if (v_verbose_3091_ == 0)
{
v_a_3099_ = v___x_3147_;
goto v___jp_3098_;
}
else
{
lean_object* v___x_3148_; 
lean_inc(v_declName_3121_);
lean_inc_ref(v___x_3092_);
v___x_3148_ = l_Lean_Environment_find_x3f(v___x_3092_, v_declName_3121_, v___x_3108_);
if (lean_obj_tag(v___x_3148_) == 0)
{
lean_object* v___x_3149_; 
v___x_3149_ = l_Lean_instInhabitedConstantInfo_default;
v___y_3111_ = v___x_3147_;
v___y_3112_ = v___x_3149_;
goto v___jp_3110_;
}
else
{
lean_object* v_val_3150_; 
v_val_3150_ = lean_ctor_get(v___x_3148_, 0);
lean_inc(v_val_3150_);
lean_dec_ref_known(v___x_3148_, 1);
v___y_3111_ = v___x_3147_;
v___y_3112_ = v_val_3150_;
goto v___jp_3110_;
}
}
}
}
v___jp_3098_:
{
size_t v___x_3100_; size_t v___x_3101_; 
v___x_3100_ = ((size_t)1ULL);
v___x_3101_ = lean_usize_add(v_i_3095_, v___x_3100_);
v_i_3095_ = v___x_3101_;
v_b_3096_ = v_a_3099_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg___boxed(lean_object* v_db_3155_, lean_object* v___x_3156_, lean_object* v_verbose_3157_, lean_object* v___x_3158_, lean_object* v_as_3159_, lean_object* v_sz_3160_, lean_object* v_i_3161_, lean_object* v_b_3162_, lean_object* v___y_3163_){
_start:
{
uint8_t v_verbose_boxed_3164_; size_t v_sz_boxed_3165_; size_t v_i_boxed_3166_; lean_object* v_res_3167_; 
v_verbose_boxed_3164_ = lean_unbox(v_verbose_3157_);
v_sz_boxed_3165_ = lean_unbox_usize(v_sz_3160_);
lean_dec(v_sz_3160_);
v_i_boxed_3166_ = lean_unbox_usize(v_i_3161_);
lean_dec(v_i_3161_);
v_res_3167_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg(v_db_3155_, v___x_3156_, v_verbose_boxed_3164_, v___x_3158_, v_as_3159_, v_sz_boxed_3165_, v_i_boxed_3166_, v_b_3162_);
lean_dec_ref(v_as_3159_);
lean_dec(v___x_3156_);
lean_dec(v_db_3155_);
return v_res_3167_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__0(void){
_start:
{
lean_object* v___x_3168_; lean_object* v___x_3169_; 
v___x_3168_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_Database_url___closed__8));
v___x_3169_ = l_Lean_stringToMessageData(v___x_3168_);
return v___x_3169_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__1(void){
_start:
{
lean_object* v___x_3170_; lean_object* v___x_3171_; lean_object* v___x_3172_; lean_object* v___x_3173_; 
v___x_3170_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__0, &lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__0_once, _init_lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__0);
v___x_3171_ = lean_unsigned_to_nat(1u);
v___x_3172_ = lean_mk_empty_array_with_capacity(v___x_3171_);
v___x_3173_ = lean_array_push(v___x_3172_, v___x_3170_);
return v___x_3173_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__4(void){
_start:
{
lean_object* v___x_3177_; lean_object* v___x_3178_; 
v___x_3177_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__3));
v___x_3178_ = l_Lean_MessageData_ofFormat(v___x_3177_);
return v___x_3178_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__7(void){
_start:
{
lean_object* v___x_3182_; lean_object* v___x_3183_; 
v___x_3182_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__6));
v___x_3183_ = l_Lean_MessageData_ofFormat(v___x_3182_);
return v___x_3183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs(lean_object* v_db_3186_, uint8_t v_verbose_3187_, lean_object* v_a_3188_, lean_object* v_a_3189_){
_start:
{
lean_object* v___x_3191_; lean_object* v_env_3192_; lean_object* v___y_3194_; lean_object* v___x_3217_; lean_object* v___x_3218_; lean_object* v___x_3219_; lean_object* v___x_3220_; uint8_t v___x_3221_; 
v___x_3191_ = lean_st_ref_get(v_a_3189_);
v_env_3192_ = lean_ctor_get(v___x_3191_, 0);
lean_inc_ref_n(v_env_3192_, 2);
lean_dec(v___x_3191_);
v___x_3217_ = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Lean_Environment_getSortedCrossRefs(v_env_3192_);
v___x_3218_ = lean_unsigned_to_nat(0u);
v___x_3219_ = lean_array_get_size(v___x_3217_);
v___x_3220_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__8));
v___x_3221_ = lean_nat_dec_lt(v___x_3218_, v___x_3219_);
if (v___x_3221_ == 0)
{
lean_dec_ref(v___x_3217_);
v___y_3194_ = v___x_3220_;
goto v___jp_3193_;
}
else
{
uint8_t v___x_3222_; 
v___x_3222_ = lean_nat_dec_le(v___x_3219_, v___x_3219_);
if (v___x_3222_ == 0)
{
if (v___x_3221_ == 0)
{
lean_dec_ref(v___x_3217_);
v___y_3194_ = v___x_3220_;
goto v___jp_3193_;
}
else
{
size_t v___x_3223_; size_t v___x_3224_; lean_object* v___x_3225_; 
v___x_3223_ = ((size_t)0ULL);
v___x_3224_ = lean_usize_of_nat(v___x_3219_);
v___x_3225_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_CrossRef_traceCrossRefs_spec__2(v_db_3186_, v___x_3217_, v___x_3223_, v___x_3224_, v___x_3220_);
lean_dec_ref(v___x_3217_);
v___y_3194_ = v___x_3225_;
goto v___jp_3193_;
}
}
else
{
size_t v___x_3226_; size_t v___x_3227_; lean_object* v___x_3228_; 
v___x_3226_ = ((size_t)0ULL);
v___x_3227_ = lean_usize_of_nat(v___x_3219_);
v___x_3228_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_CrossRef_traceCrossRefs_spec__2(v_db_3186_, v___x_3217_, v___x_3226_, v___x_3227_, v___x_3220_);
lean_dec_ref(v___x_3217_);
v___y_3194_ = v___x_3228_;
goto v___jp_3193_;
}
}
v___jp_3193_:
{
lean_object* v___x_3195_; lean_object* v___x_3196_; uint8_t v___x_3197_; 
v___x_3195_ = lean_array_get_size(v___y_3194_);
v___x_3196_ = lean_unsigned_to_nat(0u);
v___x_3197_ = lean_nat_dec_eq(v___x_3195_, v___x_3196_);
if (v___x_3197_ == 0)
{
lean_object* v___x_3198_; size_t v_sz_3199_; size_t v___x_3200_; lean_object* v___x_3201_; 
v___x_3198_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__1, &lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__1_once, _init_lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__1);
v_sz_3199_ = lean_array_size(v___y_3194_);
v___x_3200_ = ((size_t)0ULL);
v___x_3201_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg(v_db_3186_, v___x_3195_, v_verbose_3187_, v_env_3192_, v___y_3194_, v_sz_3199_, v___x_3200_, v___x_3198_);
lean_dec_ref(v___y_3194_);
if (lean_obj_tag(v___x_3201_) == 0)
{
lean_object* v_a_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; lean_object* v___x_3205_; lean_object* v___x_3206_; 
v_a_3202_ = lean_ctor_get(v___x_3201_, 0);
lean_inc(v_a_3202_);
lean_dec_ref_known(v___x_3201_, 1);
v___x_3203_ = lean_array_to_list(v_a_3202_);
v___x_3204_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__4, &lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__4_once, _init_lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__4);
v___x_3205_ = l_Lean_MessageData_joinSep(v___x_3203_, v___x_3204_);
v___x_3206_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1(v___x_3205_, v_a_3188_, v_a_3189_);
return v___x_3206_;
}
else
{
lean_object* v_a_3207_; lean_object* v___x_3209_; uint8_t v_isShared_3210_; uint8_t v_isSharedCheck_3214_; 
v_a_3207_ = lean_ctor_get(v___x_3201_, 0);
v_isSharedCheck_3214_ = !lean_is_exclusive(v___x_3201_);
if (v_isSharedCheck_3214_ == 0)
{
v___x_3209_ = v___x_3201_;
v_isShared_3210_ = v_isSharedCheck_3214_;
goto v_resetjp_3208_;
}
else
{
lean_inc(v_a_3207_);
lean_dec(v___x_3201_);
v___x_3209_ = lean_box(0);
v_isShared_3210_ = v_isSharedCheck_3214_;
goto v_resetjp_3208_;
}
v_resetjp_3208_:
{
lean_object* v___x_3212_; 
if (v_isShared_3210_ == 0)
{
v___x_3212_ = v___x_3209_;
goto v_reusejp_3211_;
}
else
{
lean_object* v_reuseFailAlloc_3213_; 
v_reuseFailAlloc_3213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3213_, 0, v_a_3207_);
v___x_3212_ = v_reuseFailAlloc_3213_;
goto v_reusejp_3211_;
}
v_reusejp_3211_:
{
return v___x_3212_;
}
}
}
}
else
{
lean_object* v___x_3215_; lean_object* v___x_3216_; 
lean_dec_ref(v___y_3194_);
lean_dec_ref(v_env_3192_);
v___x_3215_ = lean_obj_once(&lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__7, &lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__7_once, _init_lp_mathlib_Mathlib_CrossRef_traceCrossRefs___closed__7);
v___x_3216_ = lp_mathlib_Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1(v___x_3215_, v_a_3188_, v_a_3189_);
return v___x_3216_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef_traceCrossRefs___boxed(lean_object* v_db_3229_, lean_object* v_verbose_3230_, lean_object* v_a_3231_, lean_object* v_a_3232_, lean_object* v_a_3233_){
_start:
{
uint8_t v_verbose_boxed_3234_; lean_object* v_res_3235_; 
v_verbose_boxed_3234_ = lean_unbox(v_verbose_3230_);
v_res_3235_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v_db_3229_, v_verbose_boxed_3234_, v_a_3231_, v_a_3232_);
lean_dec(v_a_3232_);
lean_dec_ref(v_a_3231_);
lean_dec(v_db_3229_);
return v_res_3235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0(lean_object* v_db_3236_, lean_object* v___x_3237_, uint8_t v_verbose_3238_, lean_object* v___x_3239_, lean_object* v_as_3240_, size_t v_sz_3241_, size_t v_i_3242_, lean_object* v_b_3243_, lean_object* v___y_3244_, lean_object* v___y_3245_){
_start:
{
lean_object* v___x_3247_; 
v___x_3247_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___redArg(v_db_3236_, v___x_3237_, v_verbose_3238_, v___x_3239_, v_as_3240_, v_sz_3241_, v_i_3242_, v_b_3243_);
return v___x_3247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0___boxed(lean_object* v_db_3248_, lean_object* v___x_3249_, lean_object* v_verbose_3250_, lean_object* v___x_3251_, lean_object* v_as_3252_, lean_object* v_sz_3253_, lean_object* v_i_3254_, lean_object* v_b_3255_, lean_object* v___y_3256_, lean_object* v___y_3257_, lean_object* v___y_3258_){
_start:
{
uint8_t v_verbose_boxed_3259_; size_t v_sz_boxed_3260_; size_t v_i_boxed_3261_; lean_object* v_res_3262_; 
v_verbose_boxed_3259_ = lean_unbox(v_verbose_3250_);
v_sz_boxed_3260_ = lean_unbox_usize(v_sz_3253_);
lean_dec(v_sz_3253_);
v_i_boxed_3261_ = lean_unbox_usize(v_i_3254_);
lean_dec(v_i_3254_);
v_res_3262_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_CrossRef_traceCrossRefs_spec__0(v_db_3248_, v___x_3249_, v_verbose_boxed_3259_, v___x_3251_, v_as_3252_, v_sz_boxed_3260_, v_i_boxed_3261_, v_b_3255_, v___y_3256_, v___y_3257_);
lean_dec(v___y_3257_);
lean_dec_ref(v___y_3256_);
lean_dec_ref(v_as_3252_);
lean_dec(v___x_3249_);
lean_dec(v_db_3248_);
return v_res_3262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4(lean_object* v_msgData_3263_, lean_object* v___y_3264_, lean_object* v___y_3265_){
_start:
{
lean_object* v___x_3267_; 
v___x_3267_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___redArg(v_msgData_3263_, v___y_3265_);
return v___x_3267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4___boxed(lean_object* v_msgData_3268_, lean_object* v___y_3269_, lean_object* v___y_3270_, lean_object* v___y_3271_){
_start:
{
lean_object* v_res_3272_; 
v_res_3272_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_CrossRef_traceCrossRefs_spec__1_spec__1_spec__2_spec__4(v_msgData_3268_, v___y_3269_, v___y_3270_);
lean_dec(v___y_3270_);
lean_dec_ref(v___y_3269_);
return v_res_3272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg(){
_start:
{
lean_object* v___x_3297_; lean_object* v___x_3298_; 
v___x_3297_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2__spec__0___redArg___closed__0);
v___x_3298_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3298_, 0, v___x_3297_);
return v___x_3298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg___boxed(lean_object* v___y_3299_){
_start:
{
lean_object* v_res_3300_; 
v_res_3300_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v_res_3300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0(lean_object* v_00_u03b1_3301_, lean_object* v___y_3302_, lean_object* v___y_3303_){
_start:
{
lean_object* v___x_3305_; 
v___x_3305_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v___x_3305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___boxed(lean_object* v_00_u03b1_3306_, lean_object* v___y_3307_, lean_object* v___y_3308_, lean_object* v___y_3309_){
_start:
{
lean_object* v_res_3310_; 
v_res_3310_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0(v_00_u03b1_3306_, v___y_3307_, v___y_3308_);
lean_dec(v___y_3308_);
lean_dec_ref(v___y_3307_);
return v_res_3310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1(lean_object* v_x_3311_, lean_object* v_a_3312_, lean_object* v_a_3313_){
_start:
{
lean_object* v___x_3315_; uint8_t v___x_3316_; 
v___x_3315_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_stacksTags___closed__1));
lean_inc(v_x_3311_);
v___x_3316_ = l_Lean_Syntax_isOfKind(v_x_3311_, v___x_3315_);
if (v___x_3316_ == 0)
{
lean_object* v___x_3317_; 
lean_dec(v_x_3311_);
v___x_3317_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v___x_3317_;
}
else
{
lean_object* v___x_3318_; lean_object* v___x_3319_; lean_object* v___x_3320_; 
v___x_3318_ = lean_unsigned_to_nat(1u);
v___x_3319_ = l_Lean_Syntax_getArg(v_x_3311_, v___x_3318_);
lean_dec(v_x_3311_);
v___x_3320_ = l_Lean_Syntax_getOptional_x3f(v___x_3319_);
lean_dec(v___x_3319_);
if (lean_obj_tag(v___x_3320_) == 0)
{
lean_object* v___x_3321_; uint8_t v___x_3322_; lean_object* v___x_3323_; 
v___x_3321_ = lean_box(4);
v___x_3322_ = 0;
v___x_3323_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3321_, v___x_3322_, v_a_3312_, v_a_3313_);
return v___x_3323_;
}
else
{
lean_object* v___x_3324_; lean_object* v___x_3325_; 
lean_dec_ref_known(v___x_3320_, 1);
v___x_3324_ = lean_box(4);
v___x_3325_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3324_, v___x_3316_, v_a_3312_, v_a_3313_);
return v___x_3325_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1___boxed(lean_object* v_x_3326_, lean_object* v_a_3327_, lean_object* v_a_3328_, lean_object* v_a_3329_){
_start:
{
lean_object* v_res_3330_; 
v_res_3330_ = lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1(v_x_3326_, v_a_3327_, v_a_3328_);
lean_dec(v_a_3328_);
lean_dec_ref(v_a_3327_);
return v_res_3330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__kerodonTags__1(lean_object* v_x_3348_, lean_object* v_a_3349_, lean_object* v_a_3350_){
_start:
{
lean_object* v___x_3352_; uint8_t v___x_3353_; 
v___x_3352_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_kerodonTags___closed__1));
lean_inc(v_x_3348_);
v___x_3353_ = l_Lean_Syntax_isOfKind(v_x_3348_, v___x_3352_);
if (v___x_3353_ == 0)
{
lean_object* v___x_3354_; 
lean_dec(v_x_3348_);
v___x_3354_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v___x_3354_;
}
else
{
lean_object* v___x_3355_; lean_object* v___x_3356_; lean_object* v___x_3357_; 
v___x_3355_ = lean_unsigned_to_nat(1u);
v___x_3356_ = l_Lean_Syntax_getArg(v_x_3348_, v___x_3355_);
lean_dec(v_x_3348_);
v___x_3357_ = l_Lean_Syntax_getOptional_x3f(v___x_3356_);
lean_dec(v___x_3356_);
if (lean_obj_tag(v___x_3357_) == 0)
{
lean_object* v___x_3358_; uint8_t v___x_3359_; lean_object* v___x_3360_; 
v___x_3358_ = lean_box(1);
v___x_3359_ = 0;
v___x_3360_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3358_, v___x_3359_, v_a_3349_, v_a_3350_);
return v___x_3360_;
}
else
{
lean_object* v___x_3361_; lean_object* v___x_3362_; 
lean_dec_ref_known(v___x_3357_, 1);
v___x_3361_ = lean_box(1);
v___x_3362_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3361_, v___x_3353_, v_a_3349_, v_a_3350_);
return v___x_3362_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__kerodonTags__1___boxed(lean_object* v_x_3363_, lean_object* v_a_3364_, lean_object* v_a_3365_, lean_object* v_a_3366_){
_start:
{
lean_object* v_res_3367_; 
v_res_3367_ = lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__kerodonTags__1(v_x_3363_, v_a_3364_, v_a_3365_);
lean_dec(v_a_3365_);
lean_dec_ref(v_a_3364_);
return v_res_3367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__wikidataTags__1(lean_object* v_x_3385_, lean_object* v_a_3386_, lean_object* v_a_3387_){
_start:
{
lean_object* v___x_3389_; uint8_t v___x_3390_; 
v___x_3389_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_wikidataTags___closed__1));
lean_inc(v_x_3385_);
v___x_3390_ = l_Lean_Syntax_isOfKind(v_x_3385_, v___x_3389_);
if (v___x_3390_ == 0)
{
lean_object* v___x_3391_; 
lean_dec(v_x_3385_);
v___x_3391_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v___x_3391_;
}
else
{
lean_object* v___x_3392_; lean_object* v___x_3393_; lean_object* v___x_3394_; 
v___x_3392_ = lean_unsigned_to_nat(1u);
v___x_3393_ = l_Lean_Syntax_getArg(v_x_3385_, v___x_3392_);
lean_dec(v_x_3385_);
v___x_3394_ = l_Lean_Syntax_getOptional_x3f(v___x_3393_);
lean_dec(v___x_3393_);
if (lean_obj_tag(v___x_3394_) == 0)
{
lean_object* v___x_3395_; uint8_t v___x_3396_; lean_object* v___x_3397_; 
v___x_3395_ = lean_box(5);
v___x_3396_ = 0;
v___x_3397_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3395_, v___x_3396_, v_a_3386_, v_a_3387_);
return v___x_3397_;
}
else
{
lean_object* v___x_3398_; lean_object* v___x_3399_; 
lean_dec_ref_known(v___x_3394_, 1);
v___x_3398_ = lean_box(5);
v___x_3399_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3398_, v___x_3390_, v_a_3386_, v_a_3387_);
return v___x_3399_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__wikidataTags__1___boxed(lean_object* v_x_3400_, lean_object* v_a_3401_, lean_object* v_a_3402_, lean_object* v_a_3403_){
_start:
{
lean_object* v_res_3404_; 
v_res_3404_ = lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__wikidataTags__1(v_x_3400_, v_a_3401_, v_a_3402_);
lean_dec(v_a_3402_);
lean_dec_ref(v_a_3401_);
return v_res_3404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__lmfdbTags__1(lean_object* v_x_3422_, lean_object* v_a_3423_, lean_object* v_a_3424_){
_start:
{
lean_object* v___x_3426_; uint8_t v___x_3427_; 
v___x_3426_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_lmfdbTags___closed__1));
lean_inc(v_x_3422_);
v___x_3427_ = l_Lean_Syntax_isOfKind(v_x_3422_, v___x_3426_);
if (v___x_3427_ == 0)
{
lean_object* v___x_3428_; 
lean_dec(v_x_3422_);
v___x_3428_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v___x_3428_;
}
else
{
lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; 
v___x_3429_ = lean_unsigned_to_nat(1u);
v___x_3430_ = l_Lean_Syntax_getArg(v_x_3422_, v___x_3429_);
lean_dec(v_x_3422_);
v___x_3431_ = l_Lean_Syntax_getOptional_x3f(v___x_3430_);
lean_dec(v___x_3430_);
if (lean_obj_tag(v___x_3431_) == 0)
{
lean_object* v___x_3432_; uint8_t v___x_3433_; lean_object* v___x_3434_; 
v___x_3432_ = lean_box(2);
v___x_3433_ = 0;
v___x_3434_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3432_, v___x_3433_, v_a_3423_, v_a_3424_);
return v___x_3434_;
}
else
{
lean_object* v___x_3435_; lean_object* v___x_3436_; 
lean_dec_ref_known(v___x_3431_, 1);
v___x_3435_ = lean_box(2);
v___x_3436_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3435_, v___x_3427_, v_a_3423_, v_a_3424_);
return v___x_3436_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__lmfdbTags__1___boxed(lean_object* v_x_3437_, lean_object* v_a_3438_, lean_object* v_a_3439_, lean_object* v_a_3440_){
_start:
{
lean_object* v_res_3441_; 
v_res_3441_ = lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__lmfdbTags__1(v_x_3437_, v_a_3438_, v_a_3439_);
lean_dec(v_a_3439_);
lean_dec_ref(v_a_3438_);
return v_res_3441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__pibaseTags__1(lean_object* v_x_3473_, lean_object* v_a_3474_, lean_object* v_a_3475_){
_start:
{
lean_object* v___x_3477_; uint8_t v___x_3478_; 
v___x_3477_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_pibaseTags___closed__1));
lean_inc(v_x_3473_);
v___x_3478_ = l_Lean_Syntax_isOfKind(v_x_3473_, v___x_3477_);
if (v___x_3478_ == 0)
{
lean_object* v___x_3479_; 
lean_dec(v_x_3473_);
v___x_3479_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v___x_3479_;
}
else
{
lean_object* v___x_3480_; lean_object* v___x_3481_; lean_object* v___x_3482_; lean_object* v_topic_3483_; lean_object* v___y_3485_; lean_object* v___x_3499_; 
v___x_3480_ = lean_unsigned_to_nat(1u);
v___x_3481_ = l_Lean_Syntax_getArg(v_x_3473_, v___x_3480_);
v___x_3482_ = lean_unsigned_to_nat(3u);
v_topic_3483_ = l_Lean_Syntax_getArg(v_x_3473_, v___x_3482_);
lean_dec(v_x_3473_);
v___x_3499_ = l_Lean_Syntax_getOptional_x3f(v___x_3481_);
lean_dec(v___x_3481_);
if (lean_obj_tag(v___x_3499_) == 0)
{
lean_object* v___x_3500_; 
v___x_3500_ = lean_box(0);
v___y_3485_ = v___x_3500_;
goto v___jp_3484_;
}
else
{
lean_object* v_val_3501_; lean_object* v___x_3503_; uint8_t v_isShared_3504_; uint8_t v_isSharedCheck_3508_; 
v_val_3501_ = lean_ctor_get(v___x_3499_, 0);
v_isSharedCheck_3508_ = !lean_is_exclusive(v___x_3499_);
if (v_isSharedCheck_3508_ == 0)
{
v___x_3503_ = v___x_3499_;
v_isShared_3504_ = v_isSharedCheck_3508_;
goto v_resetjp_3502_;
}
else
{
lean_inc(v_val_3501_);
lean_dec(v___x_3499_);
v___x_3503_ = lean_box(0);
v_isShared_3504_ = v_isSharedCheck_3508_;
goto v_resetjp_3502_;
}
v_resetjp_3502_:
{
lean_object* v___x_3506_; 
if (v_isShared_3504_ == 0)
{
v___x_3506_ = v___x_3503_;
goto v_reusejp_3505_;
}
else
{
lean_object* v_reuseFailAlloc_3507_; 
v_reuseFailAlloc_3507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3507_, 0, v_val_3501_);
v___x_3506_ = v_reuseFailAlloc_3507_;
goto v_reusejp_3505_;
}
v_reusejp_3505_:
{
v___y_3485_ = v___x_3506_;
goto v___jp_3484_;
}
}
}
v___jp_3484_:
{
lean_object* v___x_3486_; 
v___x_3486_ = lp_mathlib_Mathlib_CrossRef_getPiBaseTopic_x3f(v_topic_3483_);
if (lean_obj_tag(v___x_3486_) == 1)
{
lean_object* v_val_3487_; lean_object* v___x_3489_; uint8_t v_isShared_3490_; uint8_t v_isSharedCheck_3497_; 
v_val_3487_ = lean_ctor_get(v___x_3486_, 0);
v_isSharedCheck_3497_ = !lean_is_exclusive(v___x_3486_);
if (v_isSharedCheck_3497_ == 0)
{
v___x_3489_ = v___x_3486_;
v_isShared_3490_ = v_isSharedCheck_3497_;
goto v_resetjp_3488_;
}
else
{
lean_inc(v_val_3487_);
lean_dec(v___x_3486_);
v___x_3489_ = lean_box(0);
v_isShared_3490_ = v_isSharedCheck_3497_;
goto v_resetjp_3488_;
}
v_resetjp_3488_:
{
lean_object* v___x_3492_; 
if (v_isShared_3490_ == 0)
{
lean_ctor_set_tag(v___x_3489_, 3);
v___x_3492_ = v___x_3489_;
goto v_reusejp_3491_;
}
else
{
lean_object* v_reuseFailAlloc_3496_; 
v_reuseFailAlloc_3496_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3496_, 0, v_val_3487_);
v___x_3492_ = v_reuseFailAlloc_3496_;
goto v_reusejp_3491_;
}
v_reusejp_3491_:
{
if (lean_obj_tag(v___y_3485_) == 0)
{
uint8_t v___x_3493_; lean_object* v___x_3494_; 
v___x_3493_ = 0;
v___x_3494_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3492_, v___x_3493_, v_a_3474_, v_a_3475_);
lean_dec_ref(v___x_3492_);
return v___x_3494_;
}
else
{
lean_object* v___x_3495_; 
lean_dec_ref_known(v___y_3485_, 1);
v___x_3495_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3492_, v___x_3478_, v_a_3474_, v_a_3475_);
lean_dec_ref(v___x_3492_);
return v___x_3495_;
}
}
}
}
else
{
lean_object* v___x_3498_; 
lean_dec(v___x_3486_);
lean_dec(v___y_3485_);
v___x_3498_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v___x_3498_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__pibaseTags__1___boxed(lean_object* v_x_3509_, lean_object* v_a_3510_, lean_object* v_a_3511_, lean_object* v_a_3512_){
_start:
{
lean_object* v_res_3513_; 
v_res_3513_ = lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__pibaseTags__1(v_x_3509_, v_a_3510_, v_a_3511_);
lean_dec(v_a_3511_);
lean_dec_ref(v_a_3510_);
return v_res_3513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__dlmfTags__1(lean_object* v_x_3531_, lean_object* v_a_3532_, lean_object* v_a_3533_){
_start:
{
lean_object* v___x_3535_; uint8_t v___x_3536_; 
v___x_3535_ = ((lean_object*)(lp_mathlib_Mathlib_CrossRef_dlmfTags___closed__1));
lean_inc(v_x_3531_);
v___x_3536_ = l_Lean_Syntax_isOfKind(v_x_3531_, v___x_3535_);
if (v___x_3536_ == 0)
{
lean_object* v___x_3537_; 
lean_dec(v_x_3531_);
v___x_3537_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__stacksTags__1_spec__0___redArg();
return v___x_3537_;
}
else
{
lean_object* v___x_3538_; lean_object* v___x_3539_; lean_object* v___x_3540_; 
v___x_3538_ = lean_unsigned_to_nat(1u);
v___x_3539_ = l_Lean_Syntax_getArg(v_x_3531_, v___x_3538_);
lean_dec(v_x_3531_);
v___x_3540_ = l_Lean_Syntax_getOptional_x3f(v___x_3539_);
lean_dec(v___x_3539_);
if (lean_obj_tag(v___x_3540_) == 0)
{
lean_object* v___x_3541_; uint8_t v___x_3542_; lean_object* v___x_3543_; 
v___x_3541_ = lean_box(0);
v___x_3542_ = 0;
v___x_3543_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3541_, v___x_3542_, v_a_3532_, v_a_3533_);
return v___x_3543_;
}
else
{
lean_object* v___x_3544_; lean_object* v___x_3545_; 
lean_dec_ref_known(v___x_3540_, 1);
v___x_3544_ = lean_box(0);
v___x_3545_ = lp_mathlib_Mathlib_CrossRef_traceCrossRefs(v___x_3544_, v___x_3536_, v_a_3532_, v_a_3533_);
return v___x_3545_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__dlmfTags__1___boxed(lean_object* v_x_3546_, lean_object* v_a_3547_, lean_object* v_a_3548_, lean_object* v_a_3549_){
_start:
{
lean_object* v_res_3550_; 
v_res_3550_ = lp_mathlib_Mathlib_CrossRef___aux__Mathlib__Tactic__CrossRefAttribute______elabRules__Mathlib__CrossRef__dlmfTags__1(v_x_3546_, v_a_3547_, v_a_3548_);
lean_dec(v_a_3548_);
lean_dec_ref(v_a_3547_);
return v_res_3550_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin) {
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
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__1___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__2___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__3___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__4___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__5___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__6___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__7___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__8___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__9___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__10___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__11___boxed__const__1);
lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12___boxed__const__1 = _init_lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12___boxed__const__1();
lean_mark_persistent(lp_mathlib_List_all___at___00Mathlib_CrossRef_dlmfIdFn_spec__0___closed__12___boxed__const__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3825811252____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_CrossRef_tagExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_tagExt);
lean_dec_ref(res);
lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot = _init_lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_stacksTagNoAntiquot);
lp_mathlib_Mathlib_CrossRef_stacksTagParser = _init_lp_mathlib_Mathlib_CrossRef_stacksTagParser();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_stacksTagParser);
lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot = _init_lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_wikidataIdNoAntiquot);
lp_mathlib_Mathlib_CrossRef_wikidataIdParser = _init_lp_mathlib_Mathlib_CrossRef_wikidataIdParser();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_wikidataIdParser);
lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot = _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_lmfdbIdNoAntiquot);
lp_mathlib_Mathlib_CrossRef_lmfdbIdParser = _init_lp_mathlib_Mathlib_CrossRef_lmfdbIdParser();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_lmfdbIdParser);
lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot = _init_lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_pibaseIdNoAntiquot);
lp_mathlib_Mathlib_CrossRef_pibaseIdParser = _init_lp_mathlib_Mathlib_CrossRef_pibaseIdParser();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_pibaseIdParser);
lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot = _init_lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_dlmfIdNoAntiquot);
lp_mathlib_Mathlib_CrossRef_dlmfIdParser = _init_lp_mathlib_Mathlib_CrossRef_dlmfIdParser();
lean_mark_persistent(lp_mathlib_Mathlib_CrossRef_dlmfIdParser);
lp_mathlib_Lean_Parser_Category_stacksTagDB = _init_lp_mathlib_Lean_Parser_Category_stacksTagDB();
lean_mark_persistent(lp_mathlib_Lean_Parser_Category_stacksTagDB);
res = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2143948442____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2535236365____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_3593266836____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_2958650249____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_CrossRefAttribute_0__Mathlib_CrossRef_initFn_00___x40_Mathlib_Tactic_CrossRefAttribute_1047927000____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
}
#ifdef __cplusplus
}
#endif
