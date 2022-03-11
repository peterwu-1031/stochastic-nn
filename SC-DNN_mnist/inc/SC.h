/****************************************************************************
  FileName     [ SC.h ]
  Author       [ YEN-JU (Andrew) LEE ]
  Copyright    [ Copyleft(c) 2021-present ALcom(III), EE, NTU, Taiwan ]
****************************************************************************/
#ifndef SC_H
#define SC_H

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <definition.h>

using namespace std;

class SC
{
public:
    SC() {}
    SC(vector<double> vec) : lfsr(vec){}
    ~SC() {}
    double to_bipolar(int);
    double print(bool*);
    bool* bit_gen(double);
    bool* XNOR(bool*, bool*);
    bool* MUX(bool*, bool*);
    bool* ReLU(bool*);
    bool* max_pool(bool*,bool*,bool*,bool*,bool*,bool*,bool*,bool*,bool*);
    bool* MUX_general(vector<bool*> &);
    bool**** conv2d(bool****, bool****, vector<bool*> &, short, short, short, short, short, short); //added by YEN-JU, to conduct conv2d on a bipolar sc
    bool** linear(bool**, vector<vector<float>>&, vector<float>&, vector<bool*>&, short, short, bool);
    float* linear(float*, vector<vector<float>>&, vector<float>&, short, short);
    bool** view(bool****, short, short);
    bool**** maxpool2d(bool****,short,short,short,short);
    bool* Stanh(vector<bool*>& );
    bool* Hardtanh(vector<bool*>& );

    //data member
    vector<double> lfsr;
    
};


#endif