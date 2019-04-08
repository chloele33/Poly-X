#include "relationship.h"
#include <iostream>

Relationship::Relationship(std::string l) : label(l) {}

void Relationship::to_String() {

    std::cout << "Relationships for element: " << label << std::endl;
    std::cout << "----------------------------------------" << std::endl;


    std::cout << "Polygon relationships: " << std::endl;
    std::map<std::string, std::tuple<float>>::iterator it = polyRel.begin();
    while(it != polyRel.end())
    {
        std::cout<< it->first <<" - ("<< std::get<0>(it->second) << ")" << std::endl;
        it++;
    }
    std::cout << std::endl;

    std::cout << "Segment relationships:" << std::endl;
    std::map<std::string, std::tuple<float, float>>::iterator it2 = segmentRel.begin();
    while(it2 != segmentRel.end())
    {
        std::cout<< it2->first <<" - ("<< std::get<0>(it2->second) << ", " <<
                    std::get<1>(it2->second) << ")" << std::endl;
        it2++;
    }
    std::cout << std::endl;

    std::cout << "Corner relationships:" << std::endl;
    std::map<std::string, std::tuple<float, float>>::iterator it3 = cornerRel.begin();
    while(it3 != cornerRel.end())
    {
        std::cout<< it3->first <<" - ("<< std::get<0>(it3->second) << ", " <<
                    std::get<1>(it3->second) << ")" << std::endl;
        it3++;
    }
    std::cout << std::endl;
}

PolyRelationship::PolyRelationship(std::string l, Relationship ll, Relationship ul, Relationship ur,
                                   Relationship lr, Relationship c) : label (l), lowLeft(ll), upLeft(ul),
                                   upRight(ur), lowRight(lr), center(c) {}


void PolyRelationship::to_String() {
    std::cout << "RELATIONSHIPS FOR POLYGON: " << label << std::endl;

    lowLeft.to_String();
    std::cout << std::endl << std::endl;
    upLeft.to_String();
    std::cout << std::endl << std::endl;
    upRight.to_String();
    std::cout << std::endl << std::endl;
    lowRight.to_String();
    std::cout << std::endl << std::endl;
    center.to_String();
    std::cout << std::endl << std::endl;
}
