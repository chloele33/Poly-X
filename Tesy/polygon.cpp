#include "polygon.h"

Polygon::Polygon(vec2 p, vec2 rot, float w, float h) : low_bound(p),
    rotation(rot), width(w), height(h) {}

Polygon::Polygon(vec2 pl, vec2 pu, std::string l) : low_bound(pl), upper_bound(pu),
    label(l) {
    //width = glm::distance(glm::vec2(pl.x, 0), glm::vec2(pu.x, 0));
	width = (vec2(pl[0], 0) - vec2(pu[0], 0)).Length();
    //height = glm::distance(glm::vec2(0, pl.y), glm::vec2(0, pu.y));
	height = (vec2(0, pl[1]), vec2(0, pu[1])).Length();
    //center = glm::vec2(pl.x + width / 2, pl.y + height / 2);
	center = vec2(pl[0] + width / 2, pl[1] + height / 2);

    vec2 upLeft = vec2(pl[0], pu[1]);
    vec2 lowRight = vec2(pu[0], pl[1]);

    Corner *c1 = new Corner(pl, label + "LL");
    Corner *c2 = new Corner(upLeft, label + "UL");
    Corner *c3 = new Corner(pu, label + "UR");
    Corner *c4 = new Corner(lowRight, label + "LR");
    corners.push_back(c1);
    corners.push_back(c2);
    corners.push_back(c3);
    corners.push_back(c4);

    Segment *seg1 = new Segment(pl, upLeft, label + "SegmentLeft");
    Segment *seg2 = new Segment(upLeft, pu, label + "SegmentUp");
    Segment *seg3 = new Segment(pu, lowRight, label + "SegmentRight");
    Segment *seg4 = new Segment(lowRight, pl, label + "SegmentLow");

    c1->setSeg1(seg1);
    c1->setSeg2(seg4);
    c2->setSeg1(seg2);
    c2->setSeg2(seg1);
    c3->setSeg1(seg3);
    c3->setSeg2(seg2);
    c4->setSeg1(seg4);
    c4->setSeg2(seg3);

    segments.push_back(seg1);
    segments.push_back(seg2);
    segments.push_back(seg3);
    segments.push_back(seg4);

}

float Polygon::boundaryDist(vec2 p) {
    float minDist = std::numeric_limits<int>::max();
    for (Segment *seg : segments) {
        float temp = seg->segmentDistance(p);
        if (temp < minDist) {
            minDist = temp;
        }
    }

    return minDist;
}


std::vector<vec2> Polygon::boundingBoxCoords() {
    // TODO
    return std::vector<vec2>();
}


vec2 Polygon::normalizedBoundingBoxCoords() {
    // TODO
	return vec2();
}

