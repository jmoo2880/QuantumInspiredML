//
// Created by Zakhar on 03.04.2017.
//

#pragma once

#include <string>
#include <iostream>

using namespace std;

void printUsage()
{
    cout << endl << "Usage: ./cdec -task [arg] -act [arg] -input [filename]" << endl
         << endl
         << "== Obligatory arguments ==" << endl
         << endl
         << "-task {arg}" << endl
         << "    | dec  - centroid decomposition" << endl
         << "    | rec  - recovery of missing values" << endl
         << "    | norm - normalization of the matrix (z-score)" << endl
         << endl
         << "-act {arg}" << endl
         << "    | res     - the result of the <task>" << endl
         << "    | prec    - precision of the <task>" << endl
         << "    | runtime - runtime of the <task>" << endl
         << endl
         << "-input {str}" << endl
         << "    | file name, where to take the input matrix from" << endl
         << endl
         << "== Optional arguments ==" << endl
         << endl
         << "[-n {int}] default(0)" << endl
         << "    | amount of rows to load from the input file" << endl
         << "    | 0 - load all of them" << endl
         << endl
         << "[-m {int}] default(0)" << endl
         << "    | amount of columns to load from the input file" << endl
         << "    | 0 - load all of them" << endl
         << endl
         << "[-k {int}] default(m)" << endl
         << "    | amount of columns of truncated decomposition to keep" << endl
         << "    | 0 (dec) - will be set to be equal to m" << endl
         << "    | 0 (rec) - will be automatically detected" << endl
         << endl
         << "[-cdvar {arg}] default(lsv-base)" << endl
         << "        | lsv-base   - local sign vector (default)" << endl
         << "        | lsv-noinit - LSV with skipped init" << endl
         << "        | issv       - incremental scalable sign vector" << endl
         << "        | issv+      - ISSV+ (lifted condition of 1 update per iteration)" << endl
         << "        | issv-init  - ISSV  but with linear init" << endl
         << "        | issv+-init - ISSV+ but with linear init" << endl
         << "    | invalid value will result in a warning message and a fallback to a default algorithm" << endl
         << endl
         << "[-output {str}] default(auto)" << endl
         << "    | file name, where to store the result of <test>" << endl
         << "    | if not provided, input name with be appended with an appropriate suffix" << endl
         << endl
         << "== Other arguments (technical) ==" << endl
         << endl
         << "[-xtra {arg}] default(<none>)" << endl
         << "        | inc            - turn on incremental mode (supplying -n and -imax is obligatory)" << endl
         << "        | ilastonly      - (incremental) print only the last action of the incremental <task>" << endl
         << "        | full           - (non-incremental) export rt/prec with info about n, m" << endl
         << "        | batch, bat     - (recovery) disable sign vector caching during recovery" << endl
         << "        | recnorm        - (recovery) use z-score normalization during recovery" << endl
         << "    | turns on a special flag" << endl
         << "    | multiple flags require \"-xtra flag1 -xtra flag2 ...\"" << endl
         << endl
         << "[-istep {int}] default(1)" << endl
         << "    | (incremental-only) amount of rows to load before re-applying <action>" << endl
         << endl
         << "[-imax {int}] default(0)" << endl
         << "    | (incremental-only) maximum amount of rows to apply <action> to" << endl
         << "    |       0 - load as much as possible" << endl
         << endl;
}

enum class PAction
{
    Decomposition, Recovery, Normalization, Undefined
};

enum class PTestType
{
    Output, Precision, Runtime, Undefined
};

struct PTestParam
{
    bool Incremental, IncrementalLastOnly, UseBatchCD, UseFullOutput, RecoverNormalized;
    
    explicit PTestParam()
            : Incremental(false), IncrementalLastOnly(false),
              UseBatchCD(false), UseFullOutput(false), RecoverNormalized(false)
    { }
};

int CommandLine2(
        int argc, char *argv[],
        
        PAction &action, PTestType &test,
        PTestParam &testParams,
        
        std::string &input, std::string &output,
        
        uint64_t &n, uint64_t &m, uint64_t &k,
        uint64_t &inc, uint64_t &max,
        
        uint64_t &rcdopt, string &signVector
)
{
    string temp;
    for (int i = 1; i < argc; ++i)
    {
        temp = argv[i];
        
        // man request
        if (temp.empty())
        {
            continue;
        }
        else if (temp == "--help" || temp == "-help" || temp == "/?")
        {
            printUsage();
            return 1;
        }
            
            // Action type, base algorithm to be used
        else if (temp == "-task")
        {
            ++i;
            temp = argv[i];
            
            if (temp == "dec" || temp == "d")
            {
                action = PAction::Decomposition;
            }
            else if (temp == "rec" || temp == "r")
            {
                action = PAction::Recovery;
            }
            else if (temp == "norm" || temp == "n")
            {
                action = PAction::Normalization;
            }
            else
            {
                cout << "Unrecognized -task argument" << endl;
                printUsage();
                return EXIT_FAILURE;
            }
        }
            
            // Test type, how the algorithm is applied
        else if (temp == "-action" || temp == "-act" || temp == "-test" || temp == "-t")
        {
            ++i;
            temp = argv[i];
            
            if (temp == "res" || temp == "r" || temp == "out" || temp == "o")
            {
                test = PTestType::Output;
            }
            else if (temp == "prec" || temp == "p")
            {
                test = PTestType::Precision;
            }
            else if (temp == "runtime" || temp == "rt")
            {
                test = PTestType::Runtime;
            }
            else
            {
                cout << "Unrecognized -act argument" << endl;
                printUsage();
                return EXIT_FAILURE;
            }
        }
            
            // Additional parameters
        else if (temp == "-cdvar")
        {
            ++i;
            signVector = argv[i];
        }
        
        else if (temp == "-xtra")
        {
            ++i;
            temp = argv[i];
            
            if (temp == "inc" || temp == "i")
            {
                testParams.Incremental = true;
            }
            else if (temp == "batch" || temp == "bat")
            {
                testParams.UseBatchCD = true;
            }
            else if (temp == "full")
            {
                testParams.UseFullOutput = true;
            }
            else if (temp == "ilastonly" || temp == "ilo")
            {
                testParams.IncrementalLastOnly = true;
            }
            else if (temp == "recnorm")
            {
                testParams.RecoverNormalized = true;
            }
            else
            {
                cout << "Unrecognized -xtra argument" << endl;
                printUsage();
                return EXIT_FAILURE;
            }
        }
            
            // in/out
        else if (temp == "-input" || temp == "-in")
        {
            ++i;
            input = argv[i];
        }
        else if (temp == "-output" || temp == "-out")
        {
            ++i;
            output = argv[i];
        }
            
            // Dimensions to override defaults
        else if (temp == "-n")
        {
            ++i;
            temp = argv[i];
            
            n = static_cast<uint64_t>(stoll(temp));
        }
        else if (temp == "-m")
        {
            ++i;
            temp = argv[i];
            
            m = static_cast<uint64_t>(stoll(temp));
        }
        else if (temp == "-k")
        {
            ++i;
            temp = argv[i];
            
            k = static_cast<uint64_t>(stoll(temp));
        }
            
            // Incremental options
        else if (temp == "-istep")
        {
            ++i;
            temp = argv[i];
            
            inc = static_cast<uint64_t>(stoll(temp));
        }
        else if (temp == "-imax")
        {
            ++i;
            temp = argv[i];
            
            max = static_cast<uint64_t>(stoll(temp));
        }
            
            // Other values
        else if (temp == "-rcdopt")
        {
            ++i;
            temp = argv[i];
            
            rcdopt = static_cast<uint64_t>(stoll(temp));
        }
        
        else
        {
            cout << "Unrecognized CLI parameter" << endl;
            printUsage();
            return EXIT_FAILURE;
        }
    }
    
    return EXIT_SUCCESS;
}