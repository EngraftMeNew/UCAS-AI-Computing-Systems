// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vmatrix_vector_mult_4x4x16__Syms.h"


VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_init_sub__TOP__0(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_init_sub__TOP__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+61,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+62,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+64+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+68,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+69,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+70,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+71,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+72,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+73,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+74,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+75,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+76,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+77,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+78,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+79,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+80,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+81,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+82,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+83,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("results", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+84+i*1,0,"",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 31,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("matrix_vector_mult_4x4x16", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+61,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+62,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+64+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+68,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+69,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+70,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+71,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+72,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+73,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+74,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+75,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+76,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+77,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+78,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+79,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+80,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+81,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+82,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+83,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("results", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+84+i*1,0,"",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 31,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("ipu0", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+61,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+62,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+64+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+17,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+18+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+22+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+26,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+27,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+61,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+62,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+64+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+5+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+28,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+29+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+33+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+37,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+38,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu2", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+61,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+62,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+64+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+9+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+39,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+40+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+44+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+48,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+49,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu3", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+61,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+62,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+63,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+64+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+13+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+50,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+51+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+55+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+59,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+60,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_init_top(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_init_top\n"); );
    // Body
    Vmatrix_vector_mult_4x4x16___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vmatrix_vector_mult_4x4x16___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vmatrix_vector_mult_4x4x16___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_register(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_register\n"); );
    // Body
    tracep->addConstCb(&Vmatrix_vector_mult_4x4x16___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vmatrix_vector_mult_4x4x16___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vmatrix_vector_mult_4x4x16___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vmatrix_vector_mult_4x4x16___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_const_0\n"); );
    // Init
    Vmatrix_vector_mult_4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_vector_mult_4x4x16___024root*>(voidSelf);
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
}

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_full_0_sub_0(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_full_0\n"); );
    // Init
    Vmatrix_vector_mult_4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_vector_mult_4x4x16___024root*>(voidSelf);
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vmatrix_vector_mult_4x4x16___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vmatrix_vector_mult_4x4x16___024root__trace_full_0_sub_0(Vmatrix_vector_mult_4x4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_vector_mult_4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_vector_mult_4x4x16___024root__trace_full_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullSData(oldp+1,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[0]),16);
    bufp->fullSData(oldp+2,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[1]),16);
    bufp->fullSData(oldp+3,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[2]),16);
    bufp->fullSData(oldp+4,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu0__weights[3]),16);
    bufp->fullSData(oldp+5,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[0]),16);
    bufp->fullSData(oldp+6,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[1]),16);
    bufp->fullSData(oldp+7,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[2]),16);
    bufp->fullSData(oldp+8,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu1__weights[3]),16);
    bufp->fullSData(oldp+9,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[0]),16);
    bufp->fullSData(oldp+10,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[1]),16);
    bufp->fullSData(oldp+11,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[2]),16);
    bufp->fullSData(oldp+12,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu2__weights[3]),16);
    bufp->fullSData(oldp+13,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[0]),16);
    bufp->fullSData(oldp+14,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[1]),16);
    bufp->fullSData(oldp+15,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[2]),16);
    bufp->fullSData(oldp+16,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellinp__ipu3__weights[3]),16);
    bufp->fullIData(oldp+17,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu0__result),32);
    bufp->fullSData(oldp+18,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+19,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+20,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+21,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+22,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+23,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+24,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+25,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+26,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                            [0U]), 
                                          VL_EXTENDS_II(32,16, 
                                                        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                        [0U])) 
                              + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                               [1U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                           [1U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                              [2U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                              [2U])) 
                                    + VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__activations_reg
                                                                [3U]), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__weights_reg
                                                                [3U])))))),32);
    bufp->fullIData(oldp+27,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu0__DOT__i),32);
    bufp->fullIData(oldp+28,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu1__result),32);
    bufp->fullSData(oldp+29,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+30,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+31,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+32,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+33,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+34,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+35,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+36,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+37,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                            [0U]), 
                                          VL_EXTENDS_II(32,16, 
                                                        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                        [0U])) 
                              + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                               [1U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                           [1U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                              [2U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                              [2U])) 
                                    + VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__activations_reg
                                                                [3U]), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__weights_reg
                                                                [3U])))))),32);
    bufp->fullIData(oldp+38,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu1__DOT__i),32);
    bufp->fullIData(oldp+39,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu2__result),32);
    bufp->fullSData(oldp+40,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+41,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+42,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+43,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+44,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+45,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+46,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+47,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+48,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                            [0U]), 
                                          VL_EXTENDS_II(32,16, 
                                                        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                        [0U])) 
                              + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                               [1U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                           [1U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                              [2U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                              [2U])) 
                                    + VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__activations_reg
                                                                [3U]), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__weights_reg
                                                                [3U])))))),32);
    bufp->fullIData(oldp+49,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu2__DOT__i),32);
    bufp->fullIData(oldp+50,(vlSelf->matrix_vector_mult_4x4x16__DOT____Vcellout__ipu3__result),32);
    bufp->fullSData(oldp+51,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+52,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+53,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+54,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+55,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+56,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+57,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+58,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+59,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                            [0U]), 
                                          VL_EXTENDS_II(32,16, 
                                                        vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                        [0U])) 
                              + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                               [1U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                           [1U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                              [2U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                              [2U])) 
                                    + VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__activations_reg
                                                                [3U]), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__weights_reg
                                                                [3U])))))),32);
    bufp->fullIData(oldp+60,(vlSelf->matrix_vector_mult_4x4x16__DOT__ipu3__DOT__i),32);
    bufp->fullBit(oldp+61,(vlSelf->clk));
    bufp->fullBit(oldp+62,(vlSelf->reset));
    bufp->fullBit(oldp+63,(vlSelf->enable));
    bufp->fullSData(oldp+64,(vlSelf->activations[0]),16);
    bufp->fullSData(oldp+65,(vlSelf->activations[1]),16);
    bufp->fullSData(oldp+66,(vlSelf->activations[2]),16);
    bufp->fullSData(oldp+67,(vlSelf->activations[3]),16);
    bufp->fullSData(oldp+68,(vlSelf->weights[0U][0U]),16);
    bufp->fullSData(oldp+69,(vlSelf->weights[0U][1U]),16);
    bufp->fullSData(oldp+70,(vlSelf->weights[0U][2U]),16);
    bufp->fullSData(oldp+71,(vlSelf->weights[0U][3U]),16);
    bufp->fullSData(oldp+72,(vlSelf->weights[1U][0U]),16);
    bufp->fullSData(oldp+73,(vlSelf->weights[1U][1U]),16);
    bufp->fullSData(oldp+74,(vlSelf->weights[1U][2U]),16);
    bufp->fullSData(oldp+75,(vlSelf->weights[1U][3U]),16);
    bufp->fullSData(oldp+76,(vlSelf->weights[2U][0U]),16);
    bufp->fullSData(oldp+77,(vlSelf->weights[2U][1U]),16);
    bufp->fullSData(oldp+78,(vlSelf->weights[2U][2U]),16);
    bufp->fullSData(oldp+79,(vlSelf->weights[2U][3U]),16);
    bufp->fullSData(oldp+80,(vlSelf->weights[3U][0U]),16);
    bufp->fullSData(oldp+81,(vlSelf->weights[3U][1U]),16);
    bufp->fullSData(oldp+82,(vlSelf->weights[3U][2U]),16);
    bufp->fullSData(oldp+83,(vlSelf->weights[3U][3U]),16);
    bufp->fullIData(oldp+84,(vlSelf->results[0]),32);
    bufp->fullIData(oldp+85,(vlSelf->results[1]),32);
    bufp->fullIData(oldp+86,(vlSelf->results[2]),32);
    bufp->fullIData(oldp+87,(vlSelf->results[3]),32);
}
