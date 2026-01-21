// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vinner_product_4x16__Syms.h"


VL_ATTR_COLD void Vinner_product_4x16___024root__trace_init_sub__TOP__0(Vinner_product_4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_init_sub__TOP__0\n"); );
    // Init
    const int c = vlSymsp->__Vm_baseCode;
    // Body
    tracep->declBit(c+11,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+12,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+13,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+14+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+18+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+22,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("inner_product_4x16", VerilatedTracePrefixType::SCOPE_MODULE);
    tracep->declBit(c+11,0,"clk",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+12,0,"reset",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->declBit(c+13,0,"enable",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1);
    tracep->pushPrefix("activations", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+14+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+18+i*1,0,"",-1, VerilatedTraceSigDirection::INPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+22,0,"result",-1, VerilatedTraceSigDirection::OUTPUT, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->pushPrefix("activations_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+1+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->pushPrefix("weights_reg", VerilatedTracePrefixType::ARRAY_UNPACKED);
    for (int i = 0; i < 4; ++i) {
        tracep->declBus(c+5+i*1,0,"",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::LOGIC, true,(i+0), 15,0);
    }
    tracep->popPrefix();
    tracep->declBus(c+9,0,"dot_product",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::WIRE, VerilatedTraceSigType::LOGIC, false,-1, 31,0);
    tracep->declBus(c+10,0,"i",-1, VerilatedTraceSigDirection::NONE, VerilatedTraceSigKind::VAR, VerilatedTraceSigType::INTEGER, false,-1, 31,0);
    tracep->popPrefix();
}

VL_ATTR_COLD void Vinner_product_4x16___024root__trace_init_top(Vinner_product_4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_init_top\n"); );
    // Body
    Vinner_product_4x16___024root__trace_init_sub__TOP__0(vlSelf, tracep);
}

VL_ATTR_COLD void Vinner_product_4x16___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
VL_ATTR_COLD void Vinner_product_4x16___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vinner_product_4x16___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp);
void Vinner_product_4x16___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/);

VL_ATTR_COLD void Vinner_product_4x16___024root__trace_register(Vinner_product_4x16___024root* vlSelf, VerilatedVcd* tracep) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_register\n"); );
    // Body
    tracep->addConstCb(&Vinner_product_4x16___024root__trace_const_0, 0U, vlSelf);
    tracep->addFullCb(&Vinner_product_4x16___024root__trace_full_0, 0U, vlSelf);
    tracep->addChgCb(&Vinner_product_4x16___024root__trace_chg_0, 0U, vlSelf);
    tracep->addCleanupCb(&Vinner_product_4x16___024root__trace_cleanup, vlSelf);
}

VL_ATTR_COLD void Vinner_product_4x16___024root__trace_const_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_const_0\n"); );
    // Init
    Vinner_product_4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vinner_product_4x16___024root*>(voidSelf);
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
}

VL_ATTR_COLD void Vinner_product_4x16___024root__trace_full_0_sub_0(Vinner_product_4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp);

VL_ATTR_COLD void Vinner_product_4x16___024root__trace_full_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_full_0\n"); );
    // Init
    Vinner_product_4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vinner_product_4x16___024root*>(voidSelf);
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    Vinner_product_4x16___024root__trace_full_0_sub_0((&vlSymsp->TOP), bufp);
}

VL_ATTR_COLD void Vinner_product_4x16___024root__trace_full_0_sub_0(Vinner_product_4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vinner_product_4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vinner_product_4x16___024root__trace_full_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode);
    // Body
    bufp->fullSData(oldp+1,(vlSelf->inner_product_4x16__DOT__activations_reg[0]),16);
    bufp->fullSData(oldp+2,(vlSelf->inner_product_4x16__DOT__activations_reg[1]),16);
    bufp->fullSData(oldp+3,(vlSelf->inner_product_4x16__DOT__activations_reg[2]),16);
    bufp->fullSData(oldp+4,(vlSelf->inner_product_4x16__DOT__activations_reg[3]),16);
    bufp->fullSData(oldp+5,(vlSelf->inner_product_4x16__DOT__weights_reg[0]),16);
    bufp->fullSData(oldp+6,(vlSelf->inner_product_4x16__DOT__weights_reg[1]),16);
    bufp->fullSData(oldp+7,(vlSelf->inner_product_4x16__DOT__weights_reg[2]),16);
    bufp->fullSData(oldp+8,(vlSelf->inner_product_4x16__DOT__weights_reg[3]),16);
    bufp->fullIData(oldp+9,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                           vlSelf->inner_product_4x16__DOT__activations_reg
                                                           [0U]), 
                                         VL_EXTENDS_II(32,16, 
                                                       vlSelf->inner_product_4x16__DOT__weights_reg
                                                       [0U])) 
                             + (VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                              vlSelf->inner_product_4x16__DOT__activations_reg
                                                              [1U]), 
                                            VL_EXTENDS_II(32,16, 
                                                          vlSelf->inner_product_4x16__DOT__weights_reg
                                                          [1U])) 
                                + (VL_MULS_III(32, 
                                               VL_EXTENDS_II(32,16, 
                                                             vlSelf->inner_product_4x16__DOT__activations_reg
                                                             [2U]), 
                                               VL_EXTENDS_II(32,16, 
                                                             vlSelf->inner_product_4x16__DOT__weights_reg
                                                             [2U])) 
                                   + VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->inner_product_4x16__DOT__activations_reg
                                                               [3U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->inner_product_4x16__DOT__weights_reg
                                                               [3U])))))),32);
    bufp->fullIData(oldp+10,(vlSelf->inner_product_4x16__DOT__i),32);
    bufp->fullBit(oldp+11,(vlSelf->clk));
    bufp->fullBit(oldp+12,(vlSelf->reset));
    bufp->fullBit(oldp+13,(vlSelf->enable));
    bufp->fullSData(oldp+14,(vlSelf->activations[0]),16);
    bufp->fullSData(oldp+15,(vlSelf->activations[1]),16);
    bufp->fullSData(oldp+16,(vlSelf->activations[2]),16);
    bufp->fullSData(oldp+17,(vlSelf->activations[3]),16);
    bufp->fullSData(oldp+18,(vlSelf->weights[0]),16);
    bufp->fullSData(oldp+19,(vlSelf->weights[1]),16);
    bufp->fullSData(oldp+20,(vlSelf->weights[2]),16);
    bufp->fullSData(oldp+21,(vlSelf->weights[3]),16);
    bufp->fullIData(oldp+22,(vlSelf->result),32);
}
