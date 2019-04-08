#include "similaritymeasures.h"

SimilarityMeasures::SimilarityMeasures() {}



float SimilarityMeasures::similarity(std::tuple<float, float> v1,
                                     std::tuple<float, float> v2) {
    float result = 0.f;
    float variance = 0.5f;

    float x = std::get<0>(v1) - std::get<0>(v2);
    result += 1 / (variance * std::sqrt(2 * 3.14159)) * std::exp((-1 / 2) * std::pow(x / variance, 2));

    x = std::get<1>(v1) - std::get<1>(v2);
    result += 1 / (variance * std::sqrt(2 * 3.14159)) * std::exp((-1 / 2) * std::pow(x / variance, 2));

//    int size = std::tuple_size<decltype(v1)>::value;
//    for (int i = 0; i < size; i++) {
//        float x = std::get<i>(v1) - std::get<i>(v2);
//        std::get<0>(v1);
//        result += 1 / (variance * std::sqrt(2 * 3.14159)) * std::exp((-1 / 2) * std::pow(x / variance, 2));
//    }
    return result;
}


float SimilarityMeasures::similarity(std::tuple<float> v1, std::tuple<float> v2) {
    float result = 0.f;
    float variance = 0.5f;

    float x = std::get<0>(v1) - std::get<0>(v2);
    result += 1 / (variance * std::sqrt(2 * 3.14159)) * std::exp((-1 / 2) * std::pow(x / variance, 2));

    return result;
}


float SimilarityMeasures::pointSimilarity(Relationship r1, Relationship r2) {
    if (r1.label != r2.label) {
        return 0.f;
    }

    float result = 0.f;

    int size1 = r1.cornerRel.size();
    int size2 = r2.cornerRel.size();

    if (size1 > size2) {
        for (int i = size2 - 1; i < size1; i++) {
            r2.cornerRel["NIL"] = std::make_tuple(0.f, 0.f);
        }
    }
    if (size2 > size1) {
        for (int i = size1 - 1; i < size2; i++) {
            r1.cornerRel["NIL"] = std::make_tuple(0.f, 0.f);
        }
    }

    std::map<std::string, std::tuple<float, float>>::iterator it1 = r1.cornerRel.begin();
    while(it1 != r1.cornerRel.end())
    {
        std::map<std::string, std::tuple<float, float>>::iterator it2 = r2.cornerRel.find(it1->first);
        if (it2 != r2.cornerRel.end()) {
            result += similarity(it1->second, it2->second);
        }
        else {
            result += 0.f;
        }

        it1++;
    }

    return result;

}


float SimilarityMeasures::shapeSimilarity(PolyRelationship p1, PolyRelationship p2) {
    if (p1.label != p2.label) {
        return 0.f;
    }

    float result = 0.f;
    result += pointSimilarity(p1.lowLeft, p2.lowLeft);
    result += pointSimilarity(p1.upLeft, p2.upLeft);
    result += pointSimilarity(p1.upRight, p2.upRight);
    result += pointSimilarity(p1.lowRight, p2.lowRight);
    result += pointSimilarity(p1.center, p2.center);

    return result;
}
