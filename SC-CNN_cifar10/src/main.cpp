#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <unordered_map>
#include "../inc/SC.h"
#include "../inc/definition.h"
#include <cmath>

using namespace std;

int main(int argc,char** argv){

    vector<double> lfsr;
    bool flag_b = false;
    int vec_num = 1000;
    unordered_map<int, bool> hash;
    double ran;
    int num = 0;
    if( argc > 1) flag_b = true;
    // cout << "flag_b = " << flag_b << " " << argv[1] << endl;
    while(hash.size() < vec_num){
        ran = (double)rand() / (RAND_MAX );
        num = ran*vec_num;
        if(hash.find(num) == hash.end()){
            hash[num] = true;
            lfsr.push_back(num);
        }
    }

    SC sc(lfsr);

    srand(time(NULL));
    cout<<"program in"<<endl;

    // vector<bool*> test_vec;
    // double total;

    // for(size_t i=0; i<64; i++){
    //     ran = 2*(double)rand() / (RAND_MAX ) - 1;
    //     test_vec.push_back(sc.bit_gen(ran));
    //     total += ran;
    // }

    // cout << total << "   " << sc.print(sc.hardtanh(test_vec)) << "\n";

    // return 0;

    vector<vector<float>> layer_1_w(256, vector<float>(3072, 0));
    vector<float> layer_1_b(256, 0);
    vector<vector<float>> layer_2_w(128, vector<float>(256, 0));
    vector<float> layer_2_b(128, 0);
    vector<vector<float>> layer_3_w(64, vector<float>(128, 0));
    vector<float> layer_3_b(64, 0);
    vector<vector<float>> layer_4_w(10, vector<float>(64, 0));
    vector<float> layer_4_b(10, 0);

    vector<vector<vector<float>>> w_params{layer_1_w,layer_2_w,layer_3_w,layer_4_w};

    vector<vector<float>> b_params = {layer_1_b, layer_2_b, layer_3_b,layer_4_b};


    ifstream fin;
    int count = 0, tmp = 0;
    string str = "";
    cout << "dnn.txt" << endl;
    fin.open("dnn.txt");
    if(!fin){
        cerr << "cannot open file!";
        return 0;
    }
    for(size_t i = 0; i < 64; ++i)
    {
        for(size_t j = 0; j < 256; ++j)
        {
            fin >> str;
            w_params[0][i][j] = stof(str);
        }
    }

    for(size_t i = 0; i < 10; ++i)
    {
        for(size_t j = 0; j < 64; ++j)
        {
            fin >> str;
            w_params[1][i][j] = stof(str);
        }
    }
    // for(size_t i = 0; i < 64; ++i)
    // {
    //     for(size_t j = 0; j < 128; ++j)
    //     {
    //         fin >> str;
    //         w_params[2][i][j] = stof(str);
    //     }
    // }
    // for(size_t i = 0; i < 10; ++i)
    // {
    //     for(size_t j = 0; j < 64; ++j)
    //     {
    //         fin >> str;
    //         w_params[3][i][j] = stof(str);
    //     }
    // }
    fin.close();
    cout << "dnn.txt finished!!" << endl;
    if(!fin){
        cerr << "cannot open file!";
        return 0;
    }
    fin.open("dnn_bias.txt");
    for(size_t i = 0; i < 64; ++i)
    {
        fin >> str;
        b_params[0][i] = stof(str);
    }
    for(size_t i = 0; i < 10; ++i)
    {
        fin >> str;
        b_params[1][i] = stof(str);
    }
    // for(size_t i = 0; i < 64; ++i)
    // {
    //     fin >> str;
    //     b_params[2][i] = stof(str);
    // }
    // for(size_t i = 0; i < 10; ++i)
    // {
    //     fin >> str;
    //     b_params[3][i] = stof(str);
    // }
    fin.close();
    cout << "dnn_bias finished!!" << endl;


    vector<vector<vector<vector<float>>>> cnn_layer_1_w(6, vector<vector<vector<float>>>(3, vector<vector<float>>(3, vector<float>(3))));
    vector<float> cnn_layer_1_b(6, 0);
    vector<vector<vector<vector<float>>>> cnn_layer_2_w(16, vector<vector<vector<float>>>(6, vector<vector<float>>(3, vector<float>(3))));
    vector<float> cnn_layer_2_b(16, 0);
    vector<vector<vector<vector<float>>>> cnn_layer_3_w(16, vector<vector<vector<float>>>(16, vector<vector<float>>(3, vector<float>(3))));
    vector<float> cnn_layer_3_b(32, 0);
    vector<vector<vector<vector<float>>>> cnn_layer_4_w(32, vector<vector<vector<float>>>(32, vector<vector<float>>(3, vector<float>(3))));
    vector<float> cnn_layer_4_b(32, 0);

    vector<vector<vector<vector<vector<float>>>>> cnn_w_params{cnn_layer_1_w,cnn_layer_2_w,cnn_layer_3_w,cnn_layer_4_w};

    vector<vector<float>> cnn_b_params = {cnn_layer_1_b, cnn_layer_2_b, cnn_layer_3_b, cnn_layer_4_b};
    
    fin.open("cnn.txt");
    if(!fin){
        cerr << "cannot open file!";
        return 0;
    }
    for(size_t i = 0; i < 6; ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            for(size_t k = 0; k < 3; ++k)
            {
                for(size_t l = 0; l < 3; ++l)
                {
                    fin >> str;
                    cnn_w_params[0][i][j][k][l] = stof(str);
                }
            }
        }
    }
    for(size_t i = 0; i < 16; ++i)
    {
        for(size_t j = 0; j < 6; ++j)
        {
            for(size_t k = 0; k < 3; ++k)
            {
                for(size_t l = 0; l < 3; ++l)
                {
                    fin >> str;
                    cnn_w_params[1][i][j][k][l] = stof(str);
                }
            }
        }
    }
    for(size_t i = 0; i < 16; ++i)
    {
        for(size_t j = 0; j < 16; ++j)
        {
            for(size_t k = 0; k < 3; ++k)
            {
                for(size_t l = 0; l < 3; ++l)
                {
                    fin >> str;
                    cnn_w_params[2][i][j][k][l] = stof(str);
                }
            }
        }
    }
    // for(size_t i = 0; i < 32; ++i)
    // {
    //     for(size_t j = 0; j < 3; ++j)
    //     {
    //         for(size_t k = 0; k < 3; ++k)
    //         {
    //             fin >> str;
    //             cnn_w_params[3][i][j][k] = stof(str);
    //         }
    //     }
    // }
    fin.close();

    fin.open("cnn_bias_1.txt");
    if(!fin){
        cerr << "cannot open file cnn!";
        return 0;
    }
    for(size_t i = 0; i < 6; ++i)
    {
        fin >> str;
        cnn_b_params[0][i] = stof(str);
    }
    for(size_t i = 0; i < 16; ++i)
    {
        fin >> str;
        cnn_b_params[1][i] = stof(str);
    }
    for(size_t i = 0; i < 16; ++i)
    {
        fin >> str;
        cnn_b_params[2][i] = stof(str);
    }
    // for(size_t i = 0; i < 32; ++i)
    // {
    //     fin >> str;
    //     cnn_b_params[3][i] = stof(str);
    // }
    fin.close();
/*
    for(int i=0; i<4; ++i){

        fin.open("input/layer_" + to_string(i+1) + "_w.txt");
        if(!fin){
            cerr << "cannot open file!";
        }
        count = 0;
        tmp = w_params[i][0].size();
        while(!fin.eof()){
            fin >> str;
            if(count/tmp < w_params[i].size()){
                w_params[i][count / tmp][count % tmp] = stof(str);
            }
            count ++;
        }
        fin.close();

        fin.open("input/layer_" + to_string(i+1)+ "_b.txt");
        if(!fin){
            cerr << "cannot open file!";
        }
        count = 0;
        while(!fin.eof()){
            fin >> str;
            if(count < b_params[i].size()){
                b_params[i][count]= stof(str);
            }
            count ++;
        }
        fin.close();
    }
    */
    
    vector<float> test;
    vector<vector<float>> testVecVec;
    vector<vector<vector<float>>> testVecVecVec;
    vector<vector<vector<vector<float>>>> test_images;
    vector<int> test_labels;
    count = 0;

    fin.open("test_image.txt");

    if(!fin){
        cerr << "tests file cannot open!";
        return 0;
    }
    while(fin >> str){
        // cout << "test data " << str << endl;
        test.push_back(stof(str));
        count ++;
        if(count % 32 == 0){
            testVecVec.push_back(test);
            test.clear();
        }
        if(count % 1024 == 0){
            testVecVecVec.push_back(testVecVec);
            testVecVec.clear();
        }
        if(count == 3072){
            test_images.push_back(testVecVecVec);
            testVecVecVec.clear();
            count = 0;
            // testVecVec.push_back(test);
            // test.clear();
            // count = 0;
        }
    }

    fin.close();

    fin.open("test_label.txt");

    if(!fin){
        cerr << "tests file cannot open!";
        return 0;
    }
    count = 0;
    while(fin >> str){
        if(count < 300){
            test_labels.push_back(stoi(str));
            count ++;
        }
    }
    fin.close();


    bool***** cnn_neurons;
    cnn_neurons = new bool****[5];

    cnn_neurons[0] = new bool***[3];
    for(size_t i = 0; i < 3; ++i){
        cnn_neurons[0][i] = new bool**[32];
        for(size_t j = 0; j < 32; ++j){
            cnn_neurons[0][i][j] = new bool*[32];
            for(size_t k = 0; k < 32; ++k){
                cnn_neurons[0][i][j][k] = new bool[bit_len];
                for(size_t l = 0; l < bit_len; ++l){
                    cnn_neurons[0][i][j][k][l] = 0;
                }
            }
        }
    }

    cnn_neurons[1] = new bool***[6];
    for(size_t i = 0; i < 6; ++i){
        cnn_neurons[1][i] = new bool**[32];
        for(size_t j = 0; j < 32; ++j){
            cnn_neurons[1][i][j] = new bool*[32];
            for(size_t k = 0; k < 32; ++k){
                cnn_neurons[1][i][j][k] = new bool[bit_len];
                for(size_t l = 0; l < bit_len; ++l){
                    cnn_neurons[1][i][j][k][l] = 0;
                }
            }
        }
    }

    cnn_neurons[2] = new bool***[16];
    for(size_t i = 0; i < 16; ++i){
        cnn_neurons[2][i] = new bool**[32];
        for(size_t j = 0; j < 32; ++j){
            cnn_neurons[2][i][j] = new bool*[32];
            for(size_t k = 0; k < 32; ++k){
                cnn_neurons[2][i][j][k] = new bool[bit_len];
                for(size_t l = 0; l < bit_len; ++l){
                    cnn_neurons[2][i][j][k][l] = 0;
                }
            }
        }
    }

    cnn_neurons[3] = new bool***[32];
    for(size_t i = 0; i < 32; ++i){
        cnn_neurons[3][i] = new bool**[32];
        for(size_t j = 0; j < 32; ++j){
            cnn_neurons[3][i][j] = new bool*[32];
            for(size_t k = 0; k < 32; ++k){
                cnn_neurons[3][i][j][k] = new bool[bit_len];
                for(size_t l = 0; l < bit_len; ++l){
                    cnn_neurons[3][i][j][k][l] = 0;
                }
            }
        }
    }

    cnn_neurons[4] = new bool***[32];
    for(size_t i = 0; i < 32; ++i){
        cnn_neurons[4][i] = new bool**[32];
        for(size_t j = 0; j < 32; ++j){
            cnn_neurons[4][i][j] = new bool*[32];
            for(size_t k = 0; k < 32; ++k){
                cnn_neurons[4][i][j][k] = new bool[bit_len];
                for(size_t l = 0; l < bit_len; ++l){
                    cnn_neurons[4][i][j][k][l] = 0;
                }
            }
        }
    }

    float**** cnn_neurons_b;
    cnn_neurons_b = new float***[4];

    cnn_neurons_b[0] = new float**[3];
    for(size_t i = 0; i < 3; ++i){
        cnn_neurons_b[0][i] = new float*[32];
        for(size_t j = 0; j < 32; ++j){
            cnn_neurons_b[0][i][j] = new float[32];
            for(size_t k = 0; k < 32; ++k){
                cnn_neurons_b[0][i][j][k] = 0;
            }
        }
    }

    cnn_neurons_b[1] = new float**[16];
    for(size_t i = 0; i < 16; ++i){
        cnn_neurons_b[1][i] = new float*[32];
        for(size_t j = 0; j < 32; ++j){
            cnn_neurons_b[1][i][j] = new float[32];
            for(size_t k = 0; k < 32; ++k){
                cnn_neurons_b[1][i][j][k] = 0;
            }
        }
    }

    cnn_neurons_b[2] = new float**[16];
    for(size_t i = 0; i < 16; ++i){
        cnn_neurons_b[2][i] = new float*[16];
        for(size_t j = 0; j < 16; ++j){
            cnn_neurons_b[2][i][j] = new float[16];
            for(size_t k = 0; k < 16; ++k){
                    cnn_neurons_b[2][i][j][k] = 0;
            }
        }
    }

    cnn_neurons_b[3] = new float**[32];
    for(size_t i = 0; i < 32; ++i){
        cnn_neurons_b[3][i] = new float*[8];
        for(size_t j = 0; j < 8; ++j){
            cnn_neurons_b[3][i][j] = new float[8];
            for(size_t k = 0; k < 8; ++k){
                cnn_neurons_b[3][i][j][k] = 0;
            }
        }
    }


    bool*** fc_neurons;
    fc_neurons = new bool**[5];
    
    fc_neurons[0] = new bool*[3072];
    for(size_t i = 0; i < 3072; ++i){
        fc_neurons[0][i] = new bool[bit_len];
        for(size_t j = 0; j < bit_len; ++j){
            fc_neurons[0][i][j] = 0;
        }
    }

    fc_neurons[1] = new bool*[512];
    for(size_t i = 0; i < 512; ++i){
        fc_neurons[1][i] = new bool[bit_len];
        for(size_t j = 0; j < bit_len; ++j){
            fc_neurons[1][i][j] = 0;
        }
    }

    fc_neurons[2] = new bool*[256];
    for(size_t i = 0; i < 256; ++i){
        fc_neurons[2][i] = new bool[bit_len];
        for(size_t j = 0; j < bit_len; ++j){
            fc_neurons[2][i][j] = 0;
        }
    }

    fc_neurons[3] = new bool*[128];
    for(size_t i = 0; i < 128; ++i){
        fc_neurons[3][i] = new bool[bit_len];
        for(size_t j = 0; j < bit_len; ++j){
            fc_neurons[3][i][j] = 0;
        }
    }

    fc_neurons[4] = new bool*[10];
    for(size_t i = 0; i < 10; ++i){
        fc_neurons[4][i] = new bool[bit_len];
        for(size_t j = 0; j < bit_len; ++j){
            fc_neurons[4][i][j] = 0;
        }
    }

    float** fc_neurons_b;
    fc_neurons_b = new float*[5];
    
    fc_neurons_b[0] = new float[3072];
    for(size_t i = 0; i < 3072; ++i){
        fc_neurons_b[0][i] = 0;
    }

    fc_neurons_b[1] = new float[512];
    for(size_t i = 0; i < 512; ++i){
        fc_neurons_b[1][i] = 0;
    }

    fc_neurons_b[2] = new float[256];
    for(size_t i = 0; i < 256; ++i){
        fc_neurons_b[2][i] = 0;
    }

    fc_neurons_b[3] = new float[128];
    for(size_t i = 0; i < 128; ++i){
        fc_neurons_b[3][i] = 0;
    }

    fc_neurons_b[4] = new float[10];
    for(size_t i = 0; i < 10; ++i){
        fc_neurons_b[4][i] = 0;
    }
    
    
    
    int label = 0, max_cand = 0, correct_count = 0;
    int max_cand_b = 0, correct_count_b = 0;
    float max = 0, max_b = 0;
    vector<bool*> vec;

    // char b;
    cout << "set input " << endl;
    for(size_t i = 0; i < 300; ++i)
    {
        count = 0;
        for(size_t j = 0; j < 3; ++j)
        {
            // cout << test_images[i][j] << "\n";
            for(size_t k = 0; k < 32; ++k)
            {
                for(size_t l = 0; l < 32; ++l)
                {
                    cnn_neurons[0][j][k][l] = sc.bit_gen(test_images[i][j][k][l]);
                    if(flag_b) cnn_neurons_b[0][j][k][l] = test_images[i][j][k][l];
                    // cout << "jkl = " << j << " " << k << " " << l << " " << test_images[i][j][k][l] << endl;
                    // fc_neurons[0][count] = sc.bit_gen(testVecVec[i][j]);
                    // fc_neurons_b[0][count] = testVecVec[i][j];
                    // cout << "test_images : " << testVecVec[i][j] << endl;
                    // count++;
                }
            }
        }

        // for(int k = 0; k < 784; ++k){
        //     cout << test_images[i][k] << "  " << sc.print(fc_neurons[0][k]) << "\n";
        // }
        cout << "cnn1 start" << endl;
        cnn_neurons[1] = sc.conv2d(cnn_neurons[0], cnn_w_params[0], cnn_b_params[0], vec, 32, 3, 6);
        if(flag_b) cnn_neurons_b[1] = sc.conv2d(cnn_neurons_b[0], cnn_w_params[0], cnn_b_params[0], vec, 32, 3, 6);
        cout << "cnn1 ended" << endl;
        
        
        
        if(flag_b) cnn_neurons_b[1] = sc.maxpool2d(cnn_neurons_b[1], 32, 6, 2, 2);
        cnn_neurons[1] = sc.maxpool2d(cnn_neurons[1], 32, 6, 2, 2);
        // cout << "maxpool succeed" << endl;
        if(flag_b) cnn_neurons_b[2] = sc.conv2d(cnn_neurons_b[1], cnn_w_params[1], cnn_b_params[1], vec, 16, 6, 16);
        cnn_neurons[2] = sc.conv2d(cnn_neurons[1], cnn_w_params[1], cnn_b_params[1], vec, 16, 6, 16);
        
        
        if(flag_b) cnn_neurons_b[2] = sc.maxpool2d(cnn_neurons_b[2], 16, 16, 2, 2);
        cnn_neurons[2] = sc.maxpool2d(cnn_neurons[2], 16, 16, 2, 2);

        if(flag_b) cnn_neurons_b[3] = sc.conv2d(cnn_neurons_b[2], cnn_w_params[2], cnn_b_params[2], vec, 8, 16, 16);
        cnn_neurons[3] = sc.conv2d(cnn_neurons[2], cnn_w_params[2], cnn_b_params[2], vec, 8, 16, 16);

        if(flag_b) cnn_neurons_b[3] = sc.maxpool2d(cnn_neurons_b[3], 8, 16, 2, 2);
        cnn_neurons[3] = sc.maxpool2d(cnn_neurons[3], 8, 16, 2, 2);

        if(flag_b) fc_neurons_b[0] = sc.view(cnn_neurons_b[3], 16, 4);
        fc_neurons[0] = sc.view(cnn_neurons[3], 16, 4);
        
        if(flag_b) fc_neurons_b[1] = sc.linear(fc_neurons_b[0], w_params[0], b_params[0], 256, 64, true);
        fc_neurons[1] = sc.linear(fc_neurons[0], w_params[0], b_params[0], vec, 256, 64, true);


        if(flag_b) fc_neurons_b[2] = sc.linear(fc_neurons_b[1], w_params[1], b_params[1], 64, 10, false);
        fc_neurons[2] = sc.linear(fc_neurons[1], w_params[1], b_params[1], vec, 64, 10, false);
        // fc_neurons_b[2] = sc.linear(fc_neurons_b[1], w_params[1], b_params[1], 128, 128, true);

        // cout << 2 << "\n";

        // for(int k = 0; k < 256; ++k){
        //     cout << fc_neurons[2][k] << "\n";
        // }
        
        // fc_neurons[3] = sc.linear(fc_neurons[2], w_params[2], b_params[2], vec, 128, 64, true);
        // fc_neurons_b[3] = sc.linear(fc_neurons_b[2], w_params[2], b_params[2], 128, 64, true);

        // cout << 3 << "\n";

        // for(int k = 0; k < 128; ++k){
        //     cout << sc.print(fc_neurons[3][k]) << "\n";
        // }

        // fc_neurons[4] = sc.linear(fc_neurons[3], w_params[3], b_params[3], vec, 64, 10, false);
        // fc_neurons_b[4] = sc.linear(fc_neurons_b[3], w_params[3], b_params[3], 64, 10, false);

        cout << 4 << "\n";

        for(int k = 0; k < 10; ++k){
            cout << sc.print(fc_neurons[2][k]) << "\n";
            if(flag_b) cout << "b " << fc_neurons_b[2][k] << "\n";
        }


        max = sc.print(fc_neurons[4][0]);
        max_cand = 0;
        if(flag_b) max_b = fc_neurons_b[2][0];
        max_cand_b = 0;
        for(size_t j = 0; j < 10; ++j)
        {
            if(sc.print(fc_neurons[2][j]) > max)
            {
                max = sc.print(fc_neurons[2][j]);
                max_cand = j;
            }
            if(flag_b && fc_neurons_b[2][j] > max_b)
            {
                max_b = fc_neurons_b[2][j];
                max_cand_b = j;
            }
        }

        cout << "label is : " << test_labels[i] << " ;predict is : " << max_cand << endl;
        if(flag_b) cout << "label is :(b) " << test_labels[i] << " ;predict is : " << max_cand_b << endl;
        if(max_cand == test_labels[i])
        {
            ++correct_count;
            // cout << "correct!!!" << endl;
        }
        // else cout << "wrong!!!" << endl;
        if(flag_b && max_cand_b == test_labels[i])
        {
            ++correct_count_b;
            // cout << "correct!!!" << endl;
        }
        cout << "accuracy = " << (float)correct_count / (i + 1) << endl;
        if(flag_b) cout << "accuracy =(b) " << (float)correct_count_b / (i + 1) << endl;
    }
    
    return 0;
}

