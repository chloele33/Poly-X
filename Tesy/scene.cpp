#include "scene.h"

Scene::Scene(std::string l) : label(l) {}


std::tuple<float, float> Scene::relationship(vec2 p, Corner *c) {
    std::tuple<float, float> t = std::make_tuple(c->cornerDistance(p), c->cornerRatio(p));
    return t;
}


std::tuple<float, float> Scene::relationship(vec2 p, Segment *s) {
    std::tuple<float, float> t = std::make_tuple(s->segmentDistance(p), s->arcLength(p));
    return t;
}


std::tuple<float> Scene::relationship(vec2 p, Polygon *poly) {
    std::tuple<float> t = std::make_tuple(poly->boundaryDist(p));
    return t;
}


void Scene::calculateRelationships() {
    for (Polygon *polyMain : polygons) {
        // All the points we will get the relationships of in the polygon
        vec2 pLowLeft = polyMain->low_bound;
        vec2 pUpLeft = vec2(polyMain->low_bound[0], polyMain->upper_bound[1]);
        vec2 pUpRight = polyMain->upper_bound;
        vec2 pLowRight = vec2(polyMain->upper_bound[0], polyMain->low_bound[1]);
        vec2 pCenter = polyMain->center;

        // The relationships that will be calculated
        Relationship lowLeft = Relationship(polyMain->label + "LL");
        Relationship upLeft = Relationship(polyMain->label + "UL");
        Relationship upRight = Relationship(polyMain->label + "UR");
        Relationship lowRight = Relationship(polyMain->label + "LR");
        Relationship center = Relationship(polyMain->label + "C");

        for (Polygon *poly : polygons) {
            if (poly->label != polyMain->label) {
                // Claculate the relationship to the polygon itself
                lowLeft.polyRel[poly->label] = relationship(pLowLeft, poly);
                upLeft.polyRel[poly->label] = relationship(pUpLeft, poly);
                upRight.polyRel[poly->label] = relationship(pUpRight, poly);
                lowRight.polyRel[poly->label] = relationship(pLowRight, poly);
                center.polyRel[poly->label] = relationship(pCenter, poly);

                // Calculate the relationships to the segments
                for (Segment *s : poly->segments) {
                    lowLeft.segmentRel[s->label] = relationship(pLowLeft, s);
                    upLeft.segmentRel[s->label] = relationship(pUpLeft, s);
                    upRight.segmentRel[s->label] = relationship(pUpRight, s);
                    lowRight.segmentRel[s->label] = relationship(pLowRight, s);
                    center.segmentRel[s->label] = relationship(pCenter, s);
                }

                // Calculate the relationship to the corners
                for (Corner *c : poly->corners) {
                    lowLeft.cornerRel[c->label] = relationship(pLowLeft, c);
                    upLeft.cornerRel[c->label] = relationship(pUpLeft, c);
                    upRight.cornerRel[c->label] = relationship(pUpRight, c);
                    lowRight.cornerRel[c->label] = relationship(pLowRight, c);
                    center.cornerRel[c->label] = relationship(pCenter, c);
                }
            }
        }

        PolyRelationship mainRel = PolyRelationship(polyMain->label, lowLeft,
                                                    upLeft, upRight, lowRight, center);
        rel.push_back(mainRel);
    }
}
