﻿using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace IO.Milvus.ApiSchema;

internal sealed class GetCompactionPlansResponse
{
    /// <summary>
    /// Merge infos.
    /// </summary>
    [JsonPropertyName("mergeInfos")]
    public IList<MilvusCompactionPlan> MergeInfos { get; set; }

    /// <summary>
    /// State.
    /// </summary>
    [JsonPropertyName("state")]
    public MilvusCompactionState State { get; set; }

    [JsonPropertyName("status")]
    public ResponseStatus Status { get; set; }
}