#ifndef SIMILARITYMEASURES_H
#define SIMILARITYMEASURES_H

#include <string>
#include <tuple>
#include <vector>

#include "relationship.h"

class SimilarityMeasures
{
public:
    SimilarityMeasures();

    float shapeSimilarity(PolyRelationship p1, PolyRelationship p2);

    float pointSimilarity(Relationship r1, Relationship r2);

    float similarity(std::tuple<float, float> v1, std::tuple<float, float> v2);
    float similarity(std::tuple<float> v1, std::tuple<float> v2);

};

#endif // SIMILARITYMEASURES_H
