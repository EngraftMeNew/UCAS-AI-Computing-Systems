#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vmatrix_vector_mult_4x4x16.h"

using namespace std;

vluint64_t sim_time = 0;

bool load_config(const string& path, int& vec_size) {
    ifstream f(path);
    if (!f) { cerr << "ERROR: 无法打开config文件: " << path << endl; return false; }
    string line;
    if (!getline(f, line)) { cerr << "ERROR: config文件为空" << endl; return false; }
    try {
        vec_size = stoi(line, nullptr, 16);
        cout << "INFO: 向量维度（16进制" << line << "转换为10进制）= " << dec << vec_size << endl;
        return true;
    } catch (...) {
        cerr << "ERROR: 解析config失败" << endl;
        return false;
    }
}

bool load_16bit_data(const string& path, vector<int16_t>& data) {
    ifstream f(path);
    if (!f) { cerr << "ERROR: 无法打开文件: " << path << endl; return false; }
    string line;
    data.clear();
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string bin;
        while (ss >> bin) {
            if (bin.size() != 16) {
                cerr << "ERROR: 16位数据长度错误: " << bin.size() << "位" << endl;
                return false;
            }
            try {
                uint16_t bits = stoi(bin, nullptr, 2);
                data.push_back(static_cast<int16_t>(bits));
            } catch (...) {
                cerr << "ERROR: 无效16位数据: " << bin << endl;
                return false;
            }
        }
    }
    cout << "INFO: " << path << " 加载 " << data.size() << " 个数据" << endl;
    return true;
}

bool load_32bit_results(const string& path, vector<int32_t>& results) {
    ifstream f(path);
    if (!f) { cerr << "ERROR: 无法打开结果文件: " << path << endl; return false; }
    string line;
    results.clear();
    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string bin;
        while (ss >> bin) {
            if (bin.size() != 32) {
                cerr << "ERROR: 32位结果长度错误: " << bin.size() << "位" << endl;
                return false;
            }
            try {
                uint32_t bits = stoul(bin, nullptr, 2);
                results.push_back(static_cast<int32_t>(bits));
            } catch (...) {
                cerr << "ERROR: 无效32位结果: " << bin << endl;
                return false;
            }
        }
    }
    cout << "INFO: 结果文件加载 " << results.size() << " 个结果" << endl;
    return true;
}

int main(int argc, char**argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    cout << "\n=== 矩阵乘向量测试 ===" << endl;

    int vec_size;
    vector<int16_t> activations, weights;
    vector<int32_t> expected_results, actual_results;

    if (!load_config("/mnt/c/CODE/dlp_ex_student/data/mv/config", vec_size) ||
        !load_16bit_data("/mnt/c/CODE/dlp_ex_student/data/mv/neuron", activations) ||
        !load_16bit_data("/mnt/c/CODE/dlp_ex_student/data/mv/weight", weights) ||
        !load_32bit_results("/mnt/c/CODE/dlp_ex_student/data/mv/result", expected_results)) {
        return 1;
    }

    // 验证数据长度
    int matrix_size = vec_size * vec_size;
    if (weights.size() != matrix_size || activations.size() != vec_size || expected_results.size() != vec_size) {
        cerr << "ERROR: 数据量不匹配（矩阵=" << weights.size() << ", 向量=" << activations.size()
             << ", 预期结果=" << expected_results.size() << "）" << endl;
        return 1;
    }

    // 打印输入数据
    cout << "\n=== 输入数据 ===" << endl;
    cout << "矩阵（weights，4x4）:" << endl;
    for (int i = 0; i < vec_size; i++) {
        for (int j = 0; j < vec_size; j++) {
            cout << (int)weights[i*vec_size + j] << "\t";
        }
        cout << endl;
    }
    cout << "向量（activations）: ";
    for (size_t i = 0; i < activations.size(); i++) {
        cout << (int)activations[i];
        if (i < activations.size() - 1) cout << " ";
    }
    cout << endl;

    // 初始化DUT
    Vmatrix_vector_mult_4x4x16* dut = new Vmatrix_vector_mult_4x4x16;
    VerilatedVcdC* wave = new VerilatedVcdC;
    dut->trace(wave, 99);
    wave->open("wave_matrix_vector.vcd");

    dut->clk = 0;
    dut->reset = 1;
    dut->enable = 0;
    for (int i = 0; i < 4; i++) {
        dut->activations[i] = 0;
        for (int j = 0; j < 4; j++) {
            dut->weights[i][j] = 0;
        }
    }

    // 仿真主循环
    const int CLK_PERIOD = 2;
    bool result_captured = false;

    while (sim_time < 80) {
        if (sim_time % (CLK_PERIOD / 2) == 0) {
            dut->clk = !dut->clk;
        }

        if (sim_time == 5) {
            dut->reset = 0;
        }

        if (sim_time == 9 && dut->clk == 0) {
            // 加载数据
            for (int i = 0; i < vec_size; i++) {
                dut->activations[i] = activations[i];
                for (int j = 0; j < vec_size; j++) {
                    dut->weights[i][j] = weights[i*vec_size + j];
                }
            }
        }

        if (sim_time == 11 && dut->clk == 0) {
            dut->enable = 1;
        }

        if (sim_time == 19 && dut->clk == 0) {
            dut->enable = 0;
        }

        // 捕获结果
        if (dut->clk == 1 && dut->reset == 0 && !result_captured && sim_time > 30) {
            for (int i = 0; i < vec_size; i++) {
                actual_results.push_back(dut->results[i]);
            }
            result_captured = true;
        }

        dut->eval();
        wave->dump(sim_time);
        sim_time++;
    }

    // 清理
    wave->close();
    delete dut;
    delete wave;

    // 结果验证（按指定格式输出）
    cout << "=== 结果验证 ===" << endl;
    for (int i = 0; i < vec_size; i++) {
        cout << "结果" << i+1 << ": 实际=" << actual_results[i]
             << ", 预期=" << expected_results[i] << " ["
             << (actual_results[i] == expected_results[i] ? "通过" : "失败") << "]" << endl;
    }

    return 0;
}
