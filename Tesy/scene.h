#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <tuple>
#include "polygon.h"
#include "relationship.h"

class Scene
{
public:

    std::vector<Polygon*> polygons;
    std::vector<PolyRelationship> rel;

    std::string label;

    Scene(std::string l);


    std::tuple<float, float> relationship(vec2 p, Corner *c);
    std::tuple<float, float> relationship(vec2 p, Segment *s);

    // TODO: change to add bouding box calculations
    std::tuple<float> relationship(vec2 p, Polygon *poly);


    void calculateRelationships();
};

#endif // SCENE_H
