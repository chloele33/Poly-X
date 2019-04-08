#include <iostream>
#include "scene.h"

using namespace std;

int main()
{
    cout << "Hello World!" << endl << endl;

    Scene scene = Scene("Scene1");

    Polygon poly1 = Polygon(vec2(-8.600055, -2.314729), vec2(-3.970598, 2.314729), "floor");
    Polygon poly2 = Polygon(vec2(-5.407, 0.542248), vec2(-4.407, 1.542248), "object");

    scene.polygons.push_back(&poly1);
    scene.polygons.push_back(&poly2);


    scene.calculateRelationships();

    for (PolyRelationship rel : scene.rel) {
        rel.to_String();
    }


    return 0;
}
