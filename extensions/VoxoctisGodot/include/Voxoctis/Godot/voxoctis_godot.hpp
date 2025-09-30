// VoxoctisGodot/include/voxoctis/godot/voxoctis_godot.hpp
#pragma once
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/core/class_db.hpp>

class VoxoctisGodot : public godot::RefCounted {
    GDCLASS(VoxoctisGodot, godot::RefCounted);

protected:
    static void _bind_methods();

public:
    VoxoctisGodot() = default;
    ~VoxoctisGodot() override = default;

    void generate_mesh();
};
