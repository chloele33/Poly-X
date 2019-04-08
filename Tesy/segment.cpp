#include "segment.h"


Segment::Segment() {}

Segment::Segment(vec2 pa, vec2 pb, std::string l) : pointA(pa), pointB(pb), label(l) {}


float Segment::segmentDistance(vec2 p) {
    //float l = glm::distance2(pointA, pointB);
	float l = (pointA - pointB).SqrLength();
    if (l == 0) {
        //return glm::distance(pointA, p);
		return (pointA - p).Length();
    }

	float dot = Dot(p - pointA, pointB - pointA) / l;

    float t = std::max(0.f, std::min(1.f, dot));
    vec2 projection = pointA + t * (pointB - pointA);
    //return glm::distance(p, projection);
	return (p - projection).Length();
}


float Segment::arcLength(vec2 p) {
    //float l = glm::distance2(pointA, pointB);
	float l = (pointA - pointB).SqrLength();
    if (l == 0) {
        //return glm::distance(pointA, p);
		return (pointA - p).Length();
    }

	float dot = Dot(p - pointA, pointB - pointA) / l;

    float t = std::max(0.f, std::min(1.f, dot));
    vec2 projection = pointA + t * (pointB - pointA);

    float radius = segmentDistance(p);
    //float angle = glm::acos(glm::dot(glm::normalize(p), glm::normalize(projection)));
	float angle = std::acos(Dot(p.Normalize(), projection.Normalize()));
    // ArcLength = ( 2 * pi * radius ) * ( angle / 360 )
    float arcLengthx = (2 * 3.1416 * radius) * (angle / 360);

    //float radius2 = glm::distance(pointA, p);
	float radius2 = (pointA - p).Length();
    //float angle2 = glm::acos(glm::dot(glm::normalize(p), glm::normalize(pointA)));
	float angle2 = std::acos(Dot(p.Normalize(), pointA.Normalize()));
    float arcLengthA = (2 * 3.1416 * radius2) * (angle2 / 360);

    //float radius3 = glm::distance(pointB, p);
	float radius3 = (pointB - p).Length();
    //float angle3 = glm::acos(glm::dot(glm::normalize(p), glm::normalize(pointB)));
	float angle3 = std::acos(Dot(p.Normalize(), pointB.Normalize()));
    float arcLengthB = (2 * 3.1416 * radius3) * (angle3 / 360);

    return (arcLengthx - arcLengthA) / (arcLengthB - arcLengthA);
}
