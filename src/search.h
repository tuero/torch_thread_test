

#include "model_evaluator.h"
#include "rnd/stonesngems_base.h"
#include "types.h"

using namespace stonesngems;


struct SearchInput {
    int index;
    RNDGameState state;
    ModelEvaluator *model_evaluator;
};

bool search(const SearchInput &input);
