// res://addons/voxoctis/NativeProxy.cs
using Godot;
using System.Collections.Generic;

public sealed class NativeProxy : RefCounted
{
    private readonly RefCounted _impl;
    private readonly Dictionary<StringName, Callable> _cache = new();
    public string ClassName { get; }

    public NativeProxy(string className)
    {
        ClassName = className;
        _impl = ClassDB.Instantiate(className) as RefCounted
            ?? throw new System.InvalidOperationException($"{className} not found — is the .gdextension loaded?");

        // Optional pre-cache: reflect all methods once (if available in your Godot version)
        try
        {
            var list = ClassDB.ClassGetMethodList(new StringName(className));
            foreach (Godot.Collections.Dictionary d in list)
            {
                // Entries have a "name" key with the method name
                var sn = new StringName((string)d["name"]);
                _cache[sn] = new Callable(_impl, sn);
            }
        }
        catch
        {
            // Older API or stripped methods? No worries — we'll cache lazily in GetCallable.
        }
    }

    private Callable GetCallable(StringName method)
    {
        if (!_cache.TryGetValue(method, out var c))
        {
            c = new Callable(_impl, method);
            _cache[method] = c;
        }
        return c;
    }

    public Variant CallV(string method, params Variant[] args)
        => GetCallable(new StringName(method)).Callv(new Godot.Collections.Array(args));

    public T Call<T>(string method, params Variant[] args)
        => (T)CallV(method, args);

    protected override void Dispose(bool disposing)
    {
        _impl.Dispose();
        base.Dispose(disposing);
    }
}