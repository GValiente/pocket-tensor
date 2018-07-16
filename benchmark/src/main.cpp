#include "kf_bench.h"
#include "fd_bench.h"
#include "pt_bench.h"

int main()
{
    int its = 9;

    testKerasifyMnist(its);
    testFrugallyDeepMnist(its);
    testPocketTensorMnist(its);

    testKerasifyImdb(its);
    testFrugallyDeepImdb(its);
    testPocketTensorImdb(its);

    return 0;
}
