#include "corner.h"

Corner::Corner() {}

Corner::Corner(vec2 p, std::string l) : point(p), label(l) {}



void Corner::setSeg1(Segment *s) {
    this->seg1 = s;
}

void Corner::setSeg2(Segment *s) {
    this->seg2 = s;
}


float Corner::cornerDistance(vec2 p) {
    //return glm::distance(point, p);
	return (point - p).Length();
}


float Corner::cornerRatio(vec2 p) {
    vec2 v1 = seg1->pointA;
    if (v1 == point) {
        v1 = seg1->pointB;
    }

    vec2 v2 = seg2->pointA;
    if (v2 == point) {
        v2 = seg2->pointB;
    }

    float segAng = std::acos(Dot((p - point).Normalize(), (v1 - point).Normalize()));
    float corAng = std::acos(Dot((v2 - point).Normalize(), (v1 - point).Normalize()));

    return segAng / corAng;
}
