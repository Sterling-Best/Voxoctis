// register_types.cpp
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
// Voxoctis GDExtension Classes
#include "example_class.h"
#include "voxoctis_godot.h"  

using namespace godot;


static void initialize_voxoctis(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) return;
    ClassDB::register_class<VoxoctisGodot>();
    UtilityFunctions::print("[Voxoctis] classes registered");
}

static void uninitialize_voxoctis(ModuleInitializationLevel) {
	UtilityFunctions::print("[Voxoctis] uninitialized");
}

extern "C" GDExtensionBool GDE_EXPORT gdextension_entry(
    GDExtensionInterfaceGetProcAddress get_proc,
    const GDExtensionClassLibraryPtr lib,
    GDExtensionInitialization *init
) {
    GDExtensionBinding::InitObject io(get_proc, lib, init);
    io.register_initializer(initialize_voxoctis);
    io.register_terminator(uninitialize_voxoctis);
    io.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return io.init();
}
