#ifndef SEGMENT_H
#define SEGMENT_H

#include "vec.h"
#include <algorithm>

class Segment
{
public:

    vec2 pointA;
    vec2 pointB;

    std::string label;

    Segment();
    Segment(vec2 pa, vec2 pb, std::string l);

    float segmentDistance(vec2 p);
    float arcLength(vec2 p);
};

#endif // SEGMENT_H
