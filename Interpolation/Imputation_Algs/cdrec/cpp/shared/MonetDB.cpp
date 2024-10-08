//
// Created by zakhar on 03/03/19.
//

#include <cassert>
#include <iostream>
#include <cmath>
#include "SharedLibFunctions.h"

#include "MonetDBExtDef.h"

extern "C"
{

int
mdb_udf_recovery_multicol(void **__inputs, void **__outputs, size_t m, size_t norm, size_t opt, size_t svs)
{
    // step 1: unpack params, they are always last
    struct cudf_data_struct_int &trunc =
        *((struct cudf_data_struct_int*)__inputs[m]);
	struct cudf_data_struct_dbl &eps =
        *((struct cudf_data_struct_dbl*)__inputs[m + 1]);

    if (eps.count <= 0 || trunc.count <= 0) return -1;

    size_t truncation = static_cast<size_t>(trunc.data[0]);
    double epsilon = eps.data[0];
    
    struct cudf_data_struct_dbl &xbase =
        *((struct cudf_data_struct_dbl*)__inputs[0]);
    size_t n = xbase.count;

    if (n == 0) return 0;

    double *_data = static_cast<double *>(malloc(sizeof(double) * n * m));

    for (size_t j = 0; j < m; j++)
    {
        struct cudf_data_struct_dbl &x =
            *((struct cudf_data_struct_dbl*)__inputs[j]);
        if (x.count != n)
        {
            free(_data);
            return -2;
        }

        for (size_t i = 0; i < n; i++)
        {
            _data[m * i + j] = x.is_null(x.data[i]) ? NAN : x.data[i];
        }
    }

    recoveryOfMissingValuesParametrized(_data, n, m,
        truncation, epsilon,
        norm, opt, svs
    );
    
    for (size_t j = 0; j < m; j++)
    {
        struct cudf_data_struct_dbl *y =
            ((struct cudf_data_struct_dbl*)__outputs[j]);
        y->initialize(y, n);
        for (size_t i = 0; i < n; i++)
        {
            y->data[i] = _data[m * i + j];
        }
    }

    free(_data);
    return 0;
}

}