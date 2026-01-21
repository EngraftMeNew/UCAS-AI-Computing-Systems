// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vmatrix_mult_4x4x4x16__Syms.h"


VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_init_sub__TOP__0(Vmatrix_mult_4x4x4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_init_sub__TOP__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+276,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+277,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+278,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+279,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+280,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+281,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+282,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+283,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+284,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+285,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+286,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+287,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+288,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+289,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+290,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+291,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+292,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+293,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+294,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+295,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+296,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+297,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+298,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+299,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+300,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+301,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+302,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+303,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+304,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+305,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+306,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+307,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("results", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+308,0,"[0]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+309,0,"[1]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+310,0,"[2]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+311,0,"[3]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+312,0,"[0]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+313,0,"[1]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+314,0,"[2]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+315,0,"[3]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+316,0,"[0]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+317,0,"[1]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+318,0,"[2]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+319,0,"[3]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+320,0,"[0]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+321,0,"[1]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+322,0,"[2]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+323,0,"[3]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("matrix_mult_4x4x4x16", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+276,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+277,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+278,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+279,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+280,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+281,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+282,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+283,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+284,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+285,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+286,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+287,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+288,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+289,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+290,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+291,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+292,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+293,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+294,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+295,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+296,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+297,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+298,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+299,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+300,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+301,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+302,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+303,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+304,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+305,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+306,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+307,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("results", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+308,0,"[0]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+309,0,"[1]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+310,0,"[2]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+311,0,"[3]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+312,0,"[0]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+313,0,"[1]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+314,0,"[2]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+315,0,"[3]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+316,0,"[0]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+317,0,"[1]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+318,0,"[2]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+319,0,"[3]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+320,0,"[0]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+321,0,"[1]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+322,0,"[2]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+323,0,"[3]",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("mxv0", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+292,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+293,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+294,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+295,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+296,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+297,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+298,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+299,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+300,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+301,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+302,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+303,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+304,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+305,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+306,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+307,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("results", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+81+i*1,0,"",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 31,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("ipu0", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+5+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+85,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+86+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+90+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+94,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+95,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+9+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+96,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+97+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+101+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+105,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+106,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu2", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+13+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+107,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+108+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+112+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+116,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+117,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu3", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+17+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+118,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+119+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+123+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+127,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+128,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("mxv1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+21+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+292,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+293,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+294,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+295,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+296,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+297,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+298,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+299,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+300,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+301,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+302,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+303,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+304,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+305,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+306,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+307,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("results", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+129+i*1,0,"",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 31,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("ipu0", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+21+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+25+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+133,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+134+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+138+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+142,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+143,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+21+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+29+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+144,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+145+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+149+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+153,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+154,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu2", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+21+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+33+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+155,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+156+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+160+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+164,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+165,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu3", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+21+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+37+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+166,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+167+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+171+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+175,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+176,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("mxv2", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+41+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+292,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+293,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+294,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+295,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+296,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+297,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+298,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+299,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+300,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+301,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+302,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+303,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+304,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+305,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+306,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+307,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("results", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+177+i*1,0,"",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 31,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("ipu0", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+41+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+45+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+181,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+182+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+186+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+190,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+191,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+41+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+49+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+192,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+193+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+197+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+201,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+202,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu2", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+41+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+53+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+203,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+204+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+208+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+212,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+213,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu3", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+41+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+57+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+214,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+215+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+219+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+223,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+224,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("mxv3", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+61+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->pushPrefix("[0]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+292,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+293,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+294,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+295,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[1]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+296,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+297,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+298,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+299,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[2]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+300,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+301,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+302,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+303,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->pushPrefix("[3]", VerilatedTracePrefixType::ARRAY_UNPACKED);
    tracep->declBus(c+304,0,"[0]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+305,0,"[1]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+306,0,"[2]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->declBus(c+307,0,"[3]",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 15,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->pushPrefix("results", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+225+i*1,0,"",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 31,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("ipu0", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+61+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+65+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+229,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+230+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+234+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+238,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+239,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu1", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+61+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+69+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+240,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+241+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+245+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+249,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+250,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu2", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+61+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+73+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+251,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+252+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+256+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+260,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+261,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->pushPrefix("ipu3", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+273,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+274,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+275,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+61+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+77+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+262,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+263+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+267+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+271,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+272,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
    tracep->popPrefix();
    tracep->popPrefix();
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_init_top(Vmatrix_mult_4x4x4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_init_top\n"); );
    // Body
    Vmatrix_mult_4x4x4x16___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vmatrix_mult_4x4x4x16___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vmatrix_mult_4x4x4x16___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_register(Vmatrix_mult_4x4x4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_register\n"); );
    // Body
    tracep->addConstCb(&Vmatrix_mult_4x4x4x16___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vmatrix_mult_4x4x4x16___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vmatrix_mult_4x4x4x16___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vmatrix_mult_4x4x4x16___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_const_0\n"); );
    // Init
    Vmatrix_mult_4x4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_mult_4x4x4x16___024root*>(voidSelf);
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_full_0_sub_0(Vmatrix_mult_4x4x4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_full_0\n"); );
    // Init
    Vmatrix_mult_4x4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_mult_4x4x4x16___024root*>(voidSelf);
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vmatrix_mult_4x4x4x16___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root__trace_full_0_sub_0(Vmatrix_mult_4x4x4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_full_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullSData(oldp+1,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[0]),16);
    bufp->fullSData(oldp+2,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[1]),16);
    bufp->fullSData(oldp+3,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[2]),16);
    bufp->fullSData(oldp+4,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[3]),16);
    bufp->fullSData(oldp+5,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[0]),16);
    bufp->fullSData(oldp+6,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[1]),16);
    bufp->fullSData(oldp+7,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[2]),16);
    bufp->fullSData(oldp+8,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[3]),16);
    bufp->fullSData(oldp+9,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[0]),16);
    bufp->fullSData(oldp+10,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[1]),16);
    bufp->fullSData(oldp+11,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[2]),16);
    bufp->fullSData(oldp+12,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[3]),16);
    bufp->fullSData(oldp+13,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[0]),16);
    bufp->fullSData(oldp+14,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[1]),16);
    bufp->fullSData(oldp+15,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[2]),16);
    bufp->fullSData(oldp+16,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[3]),16);
    bufp->fullSData(oldp+17,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[0]),16);
    bufp->fullSData(oldp+18,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[1]),16);
    bufp->fullSData(oldp+19,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[2]),16);
    bufp->fullSData(oldp+20,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[3]),16);
    bufp->fullSData(oldp+21,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[0]),16);
    bufp->fullSData(oldp+22,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[1]),16);
    bufp->fullSData(oldp+23,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[2]),16);
    bufp->fullSData(oldp+24,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[3]),16);
    bufp->fullSData(oldp+25,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[0]),16);
    bufp->fullSData(oldp+26,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[1]),16);
    bufp->fullSData(oldp+27,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[2]),16);
    bufp->fullSData(oldp+28,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[3]),16);
    bufp->fullSData(oldp+29,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[0]),16);
    bufp->fullSData(oldp+30,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[1]),16);
    bufp->fullSData(oldp+31,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[2]),16);
    bufp->fullSData(oldp+32,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[3]),16);
    bufp->fullSData(oldp+33,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[0]),16);
    bufp->fullSData(oldp+34,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[1]),16);
    bufp->fullSData(oldp+35,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[2]),16);
    bufp->fullSData(oldp+36,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[3]),16);
    bufp->fullSData(oldp+37,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[0]),16);
    bufp->fullSData(oldp+38,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[1]),16);
    bufp->fullSData(oldp+39,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[2]),16);
    bufp->fullSData(oldp+40,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[3]),16);
    bufp->fullSData(oldp+41,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[0]),16);
    bufp->fullSData(oldp+42,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[1]),16);
    bufp->fullSData(oldp+43,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[2]),16);
    bufp->fullSData(oldp+44,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[3]),16);
    bufp->fullSData(oldp+45,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[0]),16);
    bufp->fullSData(oldp+46,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[1]),16);
    bufp->fullSData(oldp+47,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[2]),16);
    bufp->fullSData(oldp+48,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[3]),16);
    bufp->fullSData(oldp+49,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[0]),16);
    bufp->fullSData(oldp+50,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[1]),16);
    bufp->fullSData(oldp+51,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[2]),16);
    bufp->fullSData(oldp+52,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[3]),16);
    bufp->fullSData(oldp+53,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[0]),16);
    bufp->fullSData(oldp+54,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[1]),16);
    bufp->fullSData(oldp+55,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[2]),16);
    bufp->fullSData(oldp+56,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[3]),16);
    bufp->fullSData(oldp+57,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[0]),16);
    bufp->fullSData(oldp+58,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[1]),16);
    bufp->fullSData(oldp+59,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[2]),16);
    bufp->fullSData(oldp+60,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[3]),16);
    bufp->fullSData(oldp+61,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[0]),16);
    bufp->fullSData(oldp+62,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[1]),16);
    bufp->fullSData(oldp+63,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[2]),16);
    bufp->fullSData(oldp+64,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[3]),16);
    bufp->fullSData(oldp+65,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[0]),16);
    bufp->fullSData(oldp+66,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[1]),16);
    bufp->fullSData(oldp+67,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[2]),16);
    bufp->fullSData(oldp+68,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[3]),16);
    bufp->fullSData(oldp+69,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[0]),16);
    bufp->fullSData(oldp+70,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[1]),16);
    bufp->fullSData(oldp+71,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[2]),16);
    bufp->fullSData(oldp+72,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[3]),16);
    bufp->fullSData(oldp+73,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[0]),16);
    bufp->fullSData(oldp+74,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[1]),16);
    bufp->fullSData(oldp+75,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[2]),16);
    bufp->fullSData(oldp+76,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[3]),16);
    bufp->fullSData(oldp+77,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[0]),16);
    bufp->fullSData(oldp+78,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[1]),16);
    bufp->fullSData(oldp+79,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[2]),16);
    bufp->fullSData(oldp+80,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[3]),16);
    bufp->fullIData(oldp+81,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[0]),32);
    bufp->fullIData(oldp+82,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[1]),32);
    bufp->fullIData(oldp+83,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[2]),32);
    bufp->fullIData(oldp+84,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[3]),32);
    bufp->fullIData(oldp+85,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu0__result),32);
    bufp->fullSData(oldp+86,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+87,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+88,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+89,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+90,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+91,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+92,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+93,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+94,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg
                                                            [0U]), 
                                          VL_EXTENDS_II(32,16, 
                                                        vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg
                                                        [0U])) 
                              + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg
                                                               [1U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg
                                                           [1U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg
                                                              [2U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg
                                                              [2U])) 
                                    + VL_MULS_III(32, 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg
                                                                [3U]), 
                                                  VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg
                                                                [3U])))))),32);
    bufp->fullIData(oldp+95,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__i),32);
    bufp->fullIData(oldp+96,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu1__result),32);
    bufp->fullSData(oldp+97,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+98,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+99,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+100,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+101,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+102,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+103,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+104,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+105,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+106,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__i),32);
    bufp->fullIData(oldp+107,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu2__result),32);
    bufp->fullSData(oldp+108,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+109,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+110,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+111,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+112,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+113,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+114,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+115,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+116,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+117,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__i),32);
    bufp->fullIData(oldp+118,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu3__result),32);
    bufp->fullSData(oldp+119,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+120,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+121,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+122,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+123,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+124,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+125,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+126,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+127,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+128,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__i),32);
    bufp->fullIData(oldp+129,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[0]),32);
    bufp->fullIData(oldp+130,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[1]),32);
    bufp->fullIData(oldp+131,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[2]),32);
    bufp->fullIData(oldp+132,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[3]),32);
    bufp->fullIData(oldp+133,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu0__result),32);
    bufp->fullSData(oldp+134,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+135,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+136,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+137,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+138,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+139,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+140,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+141,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+142,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+143,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__i),32);
    bufp->fullIData(oldp+144,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu1__result),32);
    bufp->fullSData(oldp+145,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+146,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+147,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+148,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+149,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+150,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+151,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+152,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+153,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+154,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__i),32);
    bufp->fullIData(oldp+155,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu2__result),32);
    bufp->fullSData(oldp+156,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+157,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+158,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+159,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+160,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+161,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+162,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+163,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+164,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+165,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__i),32);
    bufp->fullIData(oldp+166,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu3__result),32);
    bufp->fullSData(oldp+167,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+168,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+169,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+170,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+171,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+172,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+173,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+174,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+175,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+176,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__i),32);
    bufp->fullIData(oldp+177,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[0]),32);
    bufp->fullIData(oldp+178,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[1]),32);
    bufp->fullIData(oldp+179,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[2]),32);
    bufp->fullIData(oldp+180,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[3]),32);
    bufp->fullIData(oldp+181,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu0__result),32);
    bufp->fullSData(oldp+182,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+183,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+184,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+185,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+186,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+187,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+188,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+189,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+190,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+191,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__i),32);
    bufp->fullIData(oldp+192,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu1__result),32);
    bufp->fullSData(oldp+193,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+194,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+195,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+196,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+197,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+198,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+199,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+200,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+201,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+202,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__i),32);
    bufp->fullIData(oldp+203,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu2__result),32);
    bufp->fullSData(oldp+204,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+205,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+206,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+207,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+208,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+209,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+210,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+211,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+212,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+213,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__i),32);
    bufp->fullIData(oldp+214,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu3__result),32);
    bufp->fullSData(oldp+215,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+216,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+217,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+218,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+219,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+220,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+221,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+222,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+223,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+224,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__i),32);
    bufp->fullIData(oldp+225,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[0]),32);
    bufp->fullIData(oldp+226,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[1]),32);
    bufp->fullIData(oldp+227,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[2]),32);
    bufp->fullIData(oldp+228,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[3]),32);
    bufp->fullIData(oldp+229,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu0__result),32);
    bufp->fullSData(oldp+230,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+231,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+232,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+233,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+234,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+235,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+236,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+237,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+238,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+239,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__i),32);
    bufp->fullIData(oldp+240,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu1__result),32);
    bufp->fullSData(oldp+241,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+242,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+243,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+244,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+245,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+246,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+247,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+248,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+249,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+250,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__i),32);
    bufp->fullIData(oldp+251,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu2__result),32);
    bufp->fullSData(oldp+252,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+253,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+254,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+255,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+256,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+257,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+258,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+259,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+260,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+261,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__i),32);
    bufp->fullIData(oldp+262,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu3__result),32);
    bufp->fullSData(oldp+263,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+264,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+265,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+266,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+267,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+268,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+269,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+270,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+271,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                             vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg
                                                             [0U]), 
                                           VL_EXTENDS_II(32,16, 
                                                         vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg
                                                         [0U])) 
                               + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg
                                                                [1U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg
                                                            [1U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg
                                                               [2U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg
                                                               [2U])) 
                                     + VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg
                                                                 [3U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg
                                                                 [3U])))))),32);
    bufp->fullIData(oldp+272,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__i),32);
    bufp->fullBit(oldp+273,(vlSelf->clk));
    bufp->fullBit(oldp+274,(vlSelf->reset));
    bufp->fullBit(oldp+275,(vlSelf->enable));
    bufp->fullSData(oldp+276,(vlSelf->activations[0U]
                              [0U]),16);
    bufp->fullSData(oldp+277,(vlSelf->activations[0U]
                              [1U]),16);
    bufp->fullSData(oldp+278,(vlSelf->activations[0U]
                              [2U]),16);
    bufp->fullSData(oldp+279,(vlSelf->activations[0U]
                              [3U]),16);
    bufp->fullSData(oldp+280,(vlSelf->activations[1U]
                              [0U]),16);
    bufp->fullSData(oldp+281,(vlSelf->activations[1U]
                              [1U]),16);
    bufp->fullSData(oldp+282,(vlSelf->activations[1U]
                              [2U]),16);
    bufp->fullSData(oldp+283,(vlSelf->activations[1U]
                              [3U]),16);
    bufp->fullSData(oldp+284,(vlSelf->activations[2U]
                              [0U]),16);
    bufp->fullSData(oldp+285,(vlSelf->activations[2U]
                              [1U]),16);
    bufp->fullSData(oldp+286,(vlSelf->activations[2U]
                              [2U]),16);
    bufp->fullSData(oldp+287,(vlSelf->activations[2U]
                              [3U]),16);
    bufp->fullSData(oldp+288,(vlSelf->activations[3U]
                              [0U]),16);
    bufp->fullSData(oldp+289,(vlSelf->activations[3U]
                              [1U]),16);
    bufp->fullSData(oldp+290,(vlSelf->activations[3U]
                              [2U]),16);
    bufp->fullSData(oldp+291,(vlSelf->activations[3U]
                              [3U]),16);
    bufp->fullSData(oldp+292,(vlSelf->weights[0U][0U]),16);
    bufp->fullSData(oldp+293,(vlSelf->weights[0U][1U]),16);
    bufp->fullSData(oldp+294,(vlSelf->weights[0U][2U]),16);
    bufp->fullSData(oldp+295,(vlSelf->weights[0U][3U]),16);
    bufp->fullSData(oldp+296,(vlSelf->weights[1U][0U]),16);
    bufp->fullSData(oldp+297,(vlSelf->weights[1U][1U]),16);
    bufp->fullSData(oldp+298,(vlSelf->weights[1U][2U]),16);
    bufp->fullSData(oldp+299,(vlSelf->weights[1U][3U]),16);
    bufp->fullSData(oldp+300,(vlSelf->weights[2U][0U]),16);
    bufp->fullSData(oldp+301,(vlSelf->weights[2U][1U]),16);
    bufp->fullSData(oldp+302,(vlSelf->weights[2U][2U]),16);
    bufp->fullSData(oldp+303,(vlSelf->weights[2U][3U]),16);
    bufp->fullSData(oldp+304,(vlSelf->weights[3U][0U]),16);
    bufp->fullSData(oldp+305,(vlSelf->weights[3U][1U]),16);
    bufp->fullSData(oldp+306,(vlSelf->weights[3U][2U]),16);
    bufp->fullSData(oldp+307,(vlSelf->weights[3U][3U]),16);
    bufp->fullIData(oldp+308,(vlSelf->results[0U][0U]),32);
    bufp->fullIData(oldp+309,(vlSelf->results[0U][1U]),32);
    bufp->fullIData(oldp+310,(vlSelf->results[0U][2U]),32);
    bufp->fullIData(oldp+311,(vlSelf->results[0U][3U]),32);
    bufp->fullIData(oldp+312,(vlSelf->results[1U][0U]),32);
    bufp->fullIData(oldp+313,(vlSelf->results[1U][1U]),32);
    bufp->fullIData(oldp+314,(vlSelf->results[1U][2U]),32);
    bufp->fullIData(oldp+315,(vlSelf->results[1U][3U]),32);
    bufp->fullIData(oldp+316,(vlSelf->results[2U][0U]),32);
    bufp->fullIData(oldp+317,(vlSelf->results[2U][1U]),32);
    bufp->fullIData(oldp+318,(vlSelf->results[2U][2U]),32);
    bufp->fullIData(oldp+319,(vlSelf->results[2U][3U]),32);
    bufp->fullIData(oldp+320,(vlSelf->results[3U][0U]),32);
    bufp->fullIData(oldp+321,(vlSelf->results[3U][1U]),32);
    bufp->fullIData(oldp+322,(vlSelf->results[3U][2U]),32);
    bufp->fullIData(oldp+323,(vlSelf->results[3U][3U]),32);
}
