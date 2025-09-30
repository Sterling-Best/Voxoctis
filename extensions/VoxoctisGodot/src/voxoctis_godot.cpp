#include "voxoctis/godot/voxoctis_godot.h"
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

void VoxoctisGodot::_bind_methods() {
    ClassDB::bind_method(D_METHOD("generate_mesh"), &VoxoctisGodot::generate_mesh);
}

void VoxoctisGodot::generate_mesh() {
    UtilityFunctions::print("Mesh Generated");
}
