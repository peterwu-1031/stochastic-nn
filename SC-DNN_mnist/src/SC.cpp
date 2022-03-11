#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <unordered_map>
#include "SC.h"
#include "definition.h"
using namespace std;



double SC::to_bipolar(int a){
    return 2.0 * a / bit_len - 1;
}


double SC::print(bool* a){
    int count = 0;
    for (size_t i = 0; i < bit_len; i++){
        if (a[i] == true){
            count ++;
        }
    }
    return to_bipolar(count);
    
}


bool* SC::bit_gen(double number){

    bool *bit_stream = new bool[bit_len];
    double prob = (number + 1.0) / 2;
    
    for(int i = 0; i< bit_len;i++){
        
        double r = (double)rand() / (RAND_MAX );
        if(r<prob){
            bit_stream[i] = true;
        }
        else{
            bit_stream[i] = false;
        }
    }
    
    return bit_stream;
}

// int* SC::int_SC_gen(double number){
//     int *int_stream = new int[bit_len];
//     double prob = (number + 1.0) / 2;
    
//     for(int i = 0; i< bit_len;i++){
        
//         double r = (double)rand() / (RAND_MAX );
//         if(r<prob){
//             bit_stream[i] = true;
//         }
//         else{
//             bit_stream[i] = false;
//         }
//     }
// }



bool* SC::XNOR(bool* a, bool* b){
    bool* XNOR_output = new bool[bit_len];
    int count = 0;
    for(int i = 0; i < bit_len; i++){
        if(a[i] == b[i]){
            XNOR_output[i] = true;
            count ++;
        }
        else{
            XNOR_output[i] = false;
        }
    }
    return XNOR_output;
}




bool* SC::MUX(bool* a, bool* b){

    bool *MUX_output = new bool[bit_len];
    for(size_t i = 0; i < bit_len; i++){
        
        double r = (double)rand() / (RAND_MAX + 1.0);
        if(r > 0.5){
            MUX_output[i] = a[i];
        }
        else{
            MUX_output[i] = b[i];
        }
        
    }
    if(print(MUX_output)*2 > 1){
        MUX_output = bit_gen(1);
    }
    else{
        MUX_output = bit_gen(print(MUX_output)*2);
    }
    return MUX_output;
}




bool* SC::MUX_general(vector<bool*> &bit_streams){
    bool* output = new bool[bit_len];
    float r;
    int count = 0;
    for(size_t i=0; i<bit_len; ++i){
        for(size_t j=0; j<bit_streams.size(); ++j){
            if(bit_streams[j][i]){
                count ++;
            }
        }
        if(count >= bit_streams.size()){
            output[i] = true;
            count -= bit_streams.size();
        }
        else{
            output[i] = false;
        }
    }
    return output;
}


bool**** SC::conv2d(bool**** input, bool**** filter,vector<bool*> &vec, short img_size, short in_channels, short out_channels, short kernel_size, short stride, short padding){
    vec.clear();
    //declare a 3D array(out_channel * img_size * img_size) for the output tensor
    bool**** output = new bool***[out_channels];
    for(unsigned i = 0; i < out_channels; ++i){ 
        output[i] = new bool**[img_size + 2 * padding];
        for( unsigned j = 0; j < img_size + 2; ++j){ 
            output[i][j] = new bool*[img_size + 2 * padding];
            for(unsigned k = 0; k < img_size + 2; ++k){
                output[i][j][k] = new bool[bit_len];
                output[i][j][k] = bit_gen(0);
            }
        }
    }
    
    //compute output channels
    for(unsigned i = 0; i < out_channels; ++i){
        for(unsigned j = 0; j < img_size; ++j){
            for(unsigned k = 0; k < img_size; ++k){
                vec.clear();
                for(unsigned m = 0; m < kernel_size; ++m){
                    for(unsigned n = 0; n < kernel_size; ++n){
                        for(unsigned t = 0; t < in_channels; ++t){
                            vec.push_back(XNOR(input[t][j + m][k + n],bit_gen(filter[t][i][m][n])));
                        }
                    }
                }
                output[i][j + padding][k + padding] = MUX_general(vec);
            }
        }
    }
    return output;
}


float* SC::linear(float* input, vector<vector<float>>& weight, vector<float>& bias, short in, short out){
    //new the output
    float* output = new float[out];
    for(unsigned i = 0; i < out; i++){
        output[i] = 0;
    }
    //compute the output of each neuron
    for(unsigned i = 0; i < out; ++i){ //for each output neuron
        for(unsigned j = 0; j < in; ++j){
            output[i] += input[j] * weight[i][j];
        }
        output[i] += bias[i];
    }
    return output;
}

bool** SC::linear(bool** input, vector<vector<float>>& weight, vector<float>& bias, vector<bool*>& vec, short in, short out, bool hardtanh){
    //new the output
    bool** output = new bool*[out];
    double count = 0;
    
    for(unsigned i = 0; i < out; i++){
        output[i] = new bool[bit_len];
    }
    //compute the output of each neuron
    for(unsigned i = 0; i < out; ++i){ //for each output neuron

        count = 0;
        vec.clear();
        for(unsigned j = 0; j < in; ++j){
            vec.push_back(XNOR(input[j], bit_gen(weight[i][j])));
        }
        vec.push_back(bit_gen(bias[i]));

        if(hardtanh){
            // for(size_t j = 0; j < in + 1; ++j){
            //     for(size_t k = 0; k < bit_len; ++k){
            //         if(vec[j][k]) count ++;
            //         else count --;
            //     }
            // }

            // if(count > bit_len) count = bit_len;
            // else if(count < -1* bit_len) count = -1*bit_len;


            // output[i] = bit_gen(count/bit_len);
            output[i] = Hardtanh(vec);
        }
        
        else{
            output[i] = MUX_general(vec);
        }
    }

    return output;
}


bool** SC::view(bool**** input, short channel, short input_size){
    bool** output = new bool*[channel * input_size * input_size];
    for(unsigned i = 0; i < channel * input_size * input_size; i++){
        output[i] = new bool[bit_len];
    }

    for(unsigned i = 0; i < channel; i++){
        for(unsigned j = 0; j < input_size; j++){
            for(unsigned k = 0; k < input_size; k++){
                output[i * input_size * input_size + j * input_size + k] = input[i][j + 1][k + 1];
            }
        }
    }
    return output;
}

//
bool**** SC::maxpool2d(bool**** input, short in_size, short channel, short kernal, short stride){
    //declare a 3D array(out_channel * img_size * img_size) for the output tensor
    bool**** output = new bool***[channel];
    for(unsigned i = 0; i < channel; i++){ 
        output[i] = new bool**[in_size / 2 + 2];
        for( unsigned j = 0; j < in_size / 2 + 2; j++){ 
            output[i][j] = new bool*[in_size / 2 + 2];
            for(unsigned k = 0; k < in_size / 2 + 2; k++){
                output[i][j][k] = new bool[bit_len];
                output[i][j][k] = bit_gen(0);
            }
        }
    }
    //find the max element of each kernal
    int max = 0, x = 0, y = 0, matrix[kernal][kernal];
    for( unsigned i = 0; i < in_size / 2; i++){ 
        for(unsigned j = 0; j < in_size / 2; j++){
            for(unsigned k = 0; k < kernal; k++){
                for(unsigned l = 0; l < 2; l++){
                    matrix[k][l] = 0;
                }
            }
            for(unsigned k = 0; k < kernal; k++){
                for(unsigned l = 0; l < kernal; l++){
                    for(unsigned m = 0; m < channel; m++){
                        matrix[k][l] += print(input[m][2 * i + k + 1][2 * j + l + 1]);
                    }
                }
            }
            max = 0;
            x = 0;
            y = 0;
            for(unsigned k = 0; k < kernal; k++){
                for(unsigned l = 0; l < 2; l++){
                    if(matrix[k][l] > max){
                        max = matrix[k][l];
                        x = k;
                        y = l;
                    }
                }
            }
            for(unsigned k = 0; k < channel; k++){
                output[k][i + 1][j + 1] = input[k][2 * i + x + 1][2 * j + y +1];
            }
        }
    }
    return output;
}

bool* SC::Stanh(vector<bool*>& bit_streams){
    int num = bit_streams.size();
    bool* ans = new bool[bit_len];
    int counter = 0, S = 0;
    for(size_t i=0; i<bit_len; ++i){
        S = 0;
        for(size_t j=0; j<num; ++j){
            if(bit_streams[j][i]) S++;
            else S--;
        }
        // cout << counter <<  "  " << S << "\n";
        counter += S;

        if(counter > 2*num-1) counter = 2*num-1;
        else if(counter < 0) counter = 0;

        if(counter > num-1) ans[i] = true;
        else ans[i] = false;
    }
    return ans;

}

bool* SC::Hardtanh(vector<bool*>& bit_streams){
    int num = bit_streams.size();
    bool* output = new bool[bit_len];
    int n = 20;
    float count = 0, total = 0;

    for(size_t i = 0; i < bit_len; ++i){
        // count: bipolar
        for(size_t j = 0; j < bit_streams.size(); ++j){
            if(bit_streams[j][i]) count ++;
            else count --;
        }
        total += (count + 1)/2;
        count = 0;

        if(i > n){
            if(total > 0){
                output[i] = true;
                total --;
            }
            else{
                output[i] = false;
            }
        }
    }
    // cout << total << "\n";
    for(size_t i = 0; i < n; ++i){
        if(total > 0){
            output[i] = true;
            total --;
        }
        else{
            output[i] = false;
        }
    }

    count = 0;
    for(size_t i = 0; i < bit_len; ++i){
        if(output[i]) count++;
    }
    
    return output;
}
