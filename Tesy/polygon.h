#ifndef POLYGON_H
#define POLYGON_H

#include <vector>
#include "segment.h"
#include "corner.h"

class Polygon
{
public:

    vec2 low_bound;
    vec2 upper_bound;
    vec2 center;
    //Orientation?
    vec2 rotation;
    float width;
    float height;

    std::string label;



    std::vector<Segment*> segments;
    std::vector<Corner*> corners;

    Polygon(vec2 pl, vec2 pu, std::string l);
    Polygon(vec2 p, vec2 rot, float w, float h);

    float boundaryDist(vec2 p);
    std::vector<vec2> boundingBoxCoords();
    vec2 normalizedBoundingBoxCoords();
};

#endif // POLYGON_H
