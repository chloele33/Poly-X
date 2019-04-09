#include "sceneset.h"


SceneSet::SceneSet() {}


void SceneSet::addPolyToScene(std::string label, Polygon poly) {
	if (scenes.find(label) == scenes.end()) {
		scenes[label] = Scene(label);
	}

	scenes[label].polygons.push_back(&poly);
}
