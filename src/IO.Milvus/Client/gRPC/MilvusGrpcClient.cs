﻿using Grpc.Core;
using Grpc.Net.Client;
using IO.Milvus.ApiSchema;
using IO.Milvus.Diagnostics;
using IO.Milvus.Grpc;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace IO.Milvus.Client.gRPC;

/// <summary>
/// Milvus gRPC client
/// </summary>
public partial class MilvusGrpcClient : IMilvusClient2
{
    /// <summary>
    /// The constructor for the <see cref="MilvusGrpcClient"/>
    /// </summary>
    /// <param name="endpoint"></param>
    /// <param name="port"></param>
    /// <param name="name"></param>
    /// <param name="password"></param>
    /// <param name="grpcChannel"></param>
    /// <param name="callOptions"></param>
    /// <param name="log"></param>
    public MilvusGrpcClient(
        string endpoint,
        int port = 19530,
        string name = "root",
        string password = "milvus",
        GrpcChannel grpcChannel = null,
        CallOptions? callOptions = default,
        ILogger log = null)
    {
        Verify.NotNull(endpoint, "Milvus client cannot be null or empty");

        var address = SanitizeEndpoint(endpoint,port);

        this._log = log ?? NullLogger<MilvusGrpcClient>.Instance;
        this._grpcChannel = grpcChannel ?? GrpcChannel.ForAddress(address);

        var authToken = Convert.ToBase64String(Encoding.UTF8.GetBytes($"{name}:{password}"));
        _callOptions = callOptions ?? new CallOptions(
            new Metadata()
            {
                { "authorization", authToken }
            });

        _grpcClient = new MilvusService.MilvusServiceClient(_grpcChannel);
    }

    ///<inheritdoc/>
    public async Task<bool> HealthAsync(CancellationToken cancellationToken)
    {
        _log.LogDebug("Check if connection is health");

        var response = await _grpcClient.CheckHealthAsync(new CheckHealthRequest(), _callOptions.WithCancellationToken(cancellationToken));
        if (!response.IsHealthy)
        {
            foreach (var reason in response.Reasons)
            {
                _log.LogWarning(reason);
            }
        }

        return response.IsHealthy;
    }

    ///<inheritdoc/>
    public override string ToString()
    {
        return $"{nameof(MilvusGrpcClient)}({_grpcChannel.Target})";
    }

    #region Private ===============================================================================
    private ILogger _log;
    private GrpcChannel _grpcChannel;
    private CallOptions _callOptions;
    private MilvusService.MilvusServiceClient _grpcClient;

    private static Uri SanitizeEndpoint(string endpoint, int? port)
    {
        Verify.IsValidUrl(nameof(endpoint), endpoint, false, true, false);

        UriBuilder builder = new(endpoint);
        if (port.HasValue) { builder.Port = port.Value; }

        return builder.Uri;
    }
    #endregion
}