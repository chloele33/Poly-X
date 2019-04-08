#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

class Relationship
{
public:

    std::string label;

    std::map<std::string, std::tuple<float, float>> cornerRel;
    std::map<std::string, std::tuple<float, float>> segmentRel;
    std::map<std::string, std::tuple<float>> polyRel;


    Relationship(std::string l);


    void to_String();
};


class PolyRelationship
{
public:

    std::string label;

    Relationship lowLeft;
    Relationship upLeft;
    Relationship upRight;
    Relationship lowRight;
    Relationship center;

    PolyRelationship(std::string l, Relationship ll, Relationship ul,
                     Relationship ur, Relationship lr, Relationship c);

    void to_String();
};
