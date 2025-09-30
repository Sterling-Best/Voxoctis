// res://addons/voxoctis/Voxoctis.cs
using Godot;

public sealed class Voxoctis : RefCounted
{
    private readonly NativeProxy _nativeProxy;

    private Voxoctis(NativeProxy p) => _nativeProxy = p;

    public static Voxoctis Create()
        => new Voxoctis(new NativeProxy("VoxoctisGodot"));

    // Generic escape hatch for new methods without editing this file:
    public Variant Invoke(string method, params Variant[] args) => _nativeProxy.CallV(method, args);

    protected override void Dispose(bool disposing)
    {
        _nativeProxy.Dispose();
        base.Dispose(disposing);
    }

    //Voxoctis Methods
    public void GenerateMesh() => _nativeProxy.Call("generate_mesh");


}