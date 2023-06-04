﻿using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace IO.Milvus;

/// <summary>
/// Collection Schema
/// </summary>
public sealed class CollectionSchema
{
    /// <summary>
    /// Auto id
    /// </summary>
    /// <remarks>
    /// deprecated later, keep compatible with c++ part now
    /// </remarks>
    [JsonPropertyName("autoID")]
    public bool AutoId { get; set; } = false;

    /// <summary>
    /// Collection description
    /// </summary>
    [JsonPropertyName("description")]
    public string Description { get; set; }

    /// <summary>
    /// Fields
    /// </summary>
    /// <remarks>
    /// Array of <see cref="FieldType"/>
    /// </remarks>
    [JsonPropertyName("fields")]
    public IList<FieldType> Fields { get; set; }

    /// <summary>
    /// Name
    /// </summary>
    [JsonPropertyName("name")]
    public string Name { get; set; }

    /// <summary>
    /// Return string value of <see cref="CollectionSchema"/>
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
        return $"CollectionSchema: {{{nameof(AutoId)}: {AutoId}, {nameof(Description)}, {Description}, {nameof(Fields)}: {Fields?.Count}}}";
    }
}