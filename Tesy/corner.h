#ifndef CORNER_H
#define CORNER_H

#include "segment.h"

class Corner
{
public:

    vec2 point;
    //std::vector<Segment> *adjSegments;
    Segment *seg1;
    Segment *seg2;

    std::string label;

    Corner();
    Corner(vec2 p, std::string l);

    void setSeg1(Segment *s);
    void setSeg2(Segment *s);

    float cornerDistance(vec2 p);

    float cornerRatio(vec2 p);
};

#endif // CORNER_H
