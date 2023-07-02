﻿using IO.Milvus.Client.REST;
using IO.Milvus.Diagnostics;
using System.Net.Http;
using System.Text.Json.Serialization;

namespace IO.Milvus.ApiSchema;

internal sealed class GetIndexStateRequest
{
    [JsonPropertyName("collection_name")]
    public string CollectionName { get; set; }

    [JsonPropertyName("field_name")]
    public string FieldName { get; set; }

    /// <summary>
    /// Database name
    /// </summary>
    [JsonPropertyName("db_name")]
    public string DbName { get; set; }

    public static GetIndexStateRequest Create(string collectionName, string fieldName, string dbName)
    {
        return new GetIndexStateRequest(collectionName, fieldName, dbName);
    }

    public Grpc.GetIndexStateRequest BuildGrpc()
    {
        Validate();

        return new Grpc.GetIndexStateRequest()
        {
            CollectionName = CollectionName,
            FieldName = FieldName,
            DbName = DbName
        };
    }

    public HttpRequestMessage BuildRest()
    {
        Validate();

        return HttpRequest.CreateGetRequest(
            $"{ApiVersion.V1}/state",
            payload: this
            );
    }

    public void Validate()
    {
        Verify.NotNullOrWhiteSpace(CollectionName);
        Verify.NotNullOrWhiteSpace(FieldName);
        Verify.NotNullOrWhiteSpace(DbName);
    }

    #region Private ====================================================================================
    public GetIndexStateRequest(string collectionName, string fieldName, string dbName)
    {
        CollectionName = collectionName;
        FieldName = fieldName;
        DbName = dbName;
    }
    #endregion
}