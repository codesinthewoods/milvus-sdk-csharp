﻿using IO.Milvus.Client.REST;
using IO.Milvus.Diagnostics;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json.Serialization;

namespace IO.Milvus.ApiSchema;

/// <summary>
/// Get the flush state of multiple segments
/// </summary>
internal sealed class GetFlushStateRequest:
    IValidatable,
    IRestRequest,
    IGrpcRequest<Grpc.GetFlushStateRequest>
{
    /// <summary>
    /// Segment ids
    /// </summary>
    [JsonPropertyName("segmentIDs")]
    public IList<long> SegmentIds { get; set; }

    public static GetFlushStateRequest Create(IList<long> segmentIDs)
    {
        return new GetFlushStateRequest(segmentIDs);
    }

    public HttpRequestMessage BuildRest()
    {
        this.Validate();

        return HttpRequest.CreateGetRequest(
            $"{ApiVersion.V1}/persist/state",
            payload: this
            );
    }

    public void Validate()
    {
        Verify.NotNullOrEmpty(SegmentIds, $"{nameof(SegmentIds)} Cannot be null or empty.");
    }

    public Grpc.GetFlushStateRequest BuildGrpc()
    {
        this.Validate();
        var request = new Grpc.GetFlushStateRequest();        
        request.SegmentIDs.AddRange(SegmentIds);
        return request;
    }

    public 

    #region Private =============================================
    GetFlushStateRequest(IList<long> segmentIds)
    {
        this.SegmentIds = segmentIds;
    }
    #endregion
}