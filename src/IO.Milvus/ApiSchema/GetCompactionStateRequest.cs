﻿using IO.Milvus.Client.REST;
using IO.Milvus.Diagnostics;
using System.Net.Http;
using System.Text.Json.Serialization;

namespace IO.Milvus.ApiSchema;

/// <summary>
/// Get the state of a compaction
/// </summary>
internal sealed class GetCompactionStateRequest:
    IValidatable,
    IRestRequest,
    IGrpcRequest<Grpc.GetCompactionStateRequest>
{
    /// <summary>
    /// Compaction ID
    /// </summary>
    [JsonPropertyName("compactionID")]
    public long CompactionId { get; set; }

    public static GetCompactionStateRequest Create(long compactionId)
    {
        return new GetCompactionStateRequest(compactionId);
    }

    public Grpc.GetCompactionStateRequest BuildGrpc()
    {
        this.Validate();

        return new Grpc.GetCompactionStateRequest()
        {
            CompactionID = CompactionId
        };
    }

    public HttpRequestMessage BuildRest()
    {
        this.Validate();

        return HttpRequest.CreateGetRequest(
            $"{ApiVersion.V1}/compaction/state"
        );
    }

    public void Validate()
    {
        Verify.True(CompactionId > 0, "Invalid collection id");
    }

    #region Prvate =========================================================================
    private GetCompactionStateRequest(long compactionId)
    {
        CompactionId = compactionId;
    }
    #endregion
}