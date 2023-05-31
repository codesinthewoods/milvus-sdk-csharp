using Grpc.Net.Client;
using IO.Milvus;
using IO.Milvus.ApiSchema;
using IO.Milvus.Client;
using IO.Milvus.Client.gRPC;
using IO.Milvus.Client.REST;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.SemanticKernel.AI.Embeddings;
using Microsoft.SemanticKernel.Memory;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Connectors.Memory.Milvus;

public class MilvusMemoryStore : IMemoryStore
{
    private const string EmbeddingFieldName = "embedding";
    private const string IdFieldName = "Id";
    private const string MetadataFieldName = "metadata";
    private readonly IMilvusClient _milvusClient;
    private readonly int _vectorSize;
    private readonly bool _zillizCloud;
    private readonly ILogger _log;

    /// <summary>
    /// Construct a milvus memory store.
    /// </summary>
    /// <param name="milvusClient">Milvus client.</param>
    /// <param name="logger">Logger.</param>
    public MilvusMemoryStore(
        IMilvusClient milvusClient, 
        int vectorSize, 
        bool zillizCloud = false, 
        ILogger log = null)
    {
        this._log = log ?? NullLogger<MilvusMemoryStore>.Instance;
        this._milvusClient = milvusClient;
        this._vectorSize = vectorSize;
        _zillizCloud = zillizCloud;
    }
    
    /// <summary>
    /// Construct a milvus memory store.
    /// </summary>
    /// <param name="host">Milvus server address.</param>
    /// <param name="port">Port</param>
    /// <param name="vectorSize">Vector size</param>
    /// <param name="logger">Logger</param>
    public MilvusMemoryStore(
        string host,
        int port,
        int vectorSize,
        bool zillizCloud = false,
        ILogger log = null)
    {
        this._vectorSize = vectorSize;
        this._log = log ?? NullLogger<MilvusMemoryStore>.Instance;
        this._milvusClient = new MilvusGrpcClient(host, port,log:log);
    }

    /// <summary>
    /// Construct a milvus memory store.
    /// </summary>
    /// <param name="host">Milvus memory store.</param>
    /// <param name="port">Port.</param>
    /// <param name="vectorSize">Vector size.</param>
    /// <param name="userName">Username.</param>
    /// <param name="password">Password.</param>
    /// <param name="logger">Logger.</param>
    public MilvusMemoryStore(
        string host,
        int port,
        int vectorSize,
        string userName = "root",
        string password = "milvus",
        bool zillizCloud = false,
        GrpcChannel grpcChannel = null,
        ILogger logger = null)
    {
        this._vectorSize = vectorSize;
        this._log = logger;
        this._milvusClient = new MilvusGrpcClient(host, port,userName,password,log:logger,grpcChannel:grpcChannel);
    }

    /// <summary>
    /// Construct a milvus memory store.
    /// </summary>
    /// <param name="host">Milvus memory store.</param>
    /// <param name="port">Port.</param>
    /// <param name="vectorSize">Vector size.</param>
    /// <param name="userName">Username.</param>
    /// <param name="password">Password.</param>
    /// <param name="logger">Logger.</param>
    public MilvusMemoryStore(
        string host,
        int port,
        int vectorSize,
        string userName = "root",
        string password = "milvus",
        bool zillizCloud = false,
        HttpClient httpClient = null,
        ILogger logger = null)
    {
        this._vectorSize = vectorSize;
        this._log = logger;
        this._milvusClient = new MilvusRestClient(host, port, userName, password, log: logger,httpClient:httpClient);
    }

    ///<inheritdoc/>
    public async Task CreateCollectionAsync(
        string collectionName, 
        CancellationToken cancellationToken = default)
    {
        if (!await this._milvusClient.HasCollectionAsync(
            collectionName,
            cancellationToken:cancellationToken))
        {
            await this._milvusClient.CreateCollectionAsync(collectionName,
                new FieldType[] { 
                    FieldType.CreateVarchar(IdFieldName,maxLength: 100,isPrimaryKey: true),
                    FieldType.CreateFloatVector(EmbeddingFieldName,_vectorSize),
                    FieldType.CreateVarchar(MetadataFieldName, maxLength: 10000) },
                cancellationToken: cancellationToken
                );
        }

        await BuildIndexAndLoad(collectionName,EmbeddingFieldName);
    }

    ///<inheritdoc/>
    public async Task DeleteCollectionAsync(
        string collectionName, 
        CancellationToken cancellationToken = default)
    {
        await _milvusClient.DropCollectionAsync(collectionName, cancellationToken);
    }

    ///<inheritdoc/>
    public async Task<bool> DoesCollectionExistAsync(
        string collectionName, 
        CancellationToken cancellationToken = default)
    {
        return await _milvusClient.HasCollectionAsync(collectionName, cancellationToken:cancellationToken);
    }

    ///<inheritdoc/>
    public async Task<MemoryRecord> GetAsync(
        string collectionName, 
        string key, 
        bool withEmbedding = false, 
        CancellationToken cancellationToken = default)
    {
        MilvusQueryResult result = await this._milvusClient.QueryAsync(collectionName,
            $"{IdFieldName} = {key}",
            new[] {EmbeddingFieldName,MetadataFieldName},
            cancellationToken:cancellationToken);

        if (result.FieldsData?.Any() != true)
        {
            return null;
        }

        var embeddingField = result.FieldsData[0] as FloatVectorField;
        var metadataField = result.FieldsData[1] as Field<string>;

        return MemoryRecord.FromJsonMetadata(
            metadataField.Data[0],
            new Embedding<float>(embeddingField.Data[0]));
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<MemoryRecord> GetBatchAsync(
        string collectionName, 
        IEnumerable<string> keys, 
        bool withEmbeddings = false,
        CancellationToken cancellationToken = default)
    {
        var keyList = keys.ToList();
        var keyGroup = GetKeyGroup(keyList);

        MilvusQueryResult result = await this._milvusClient.QueryAsync(collectionName,
            $"{IdFieldName} in [{keyGroup.ToString()}]",
            new[] { EmbeddingFieldName },
            cancellationToken: cancellationToken);

        if (result.FieldsData?.Any() != true)
        {
            yield break;
        }

        for (int i = 0; i < result.FieldsData.Count; i++)
        {
            var embeddingField = result.FieldsData[i] as FloatVectorField;
            var metadataField = result.FieldsData[i] as Field<string>;

            yield return MemoryRecord.FromJsonMetadata(
                metadataField.Data[i],
                new Embedding<float>(embeddingField.Data[i]));
        }
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<string> GetCollectionsAsync(CancellationToken cancellationToken = default)
    {
        var result = await _milvusClient.ShowCollectionsAsync(cancellationToken: cancellationToken);
        foreach (var collection in result)
        {
            yield return collection.CollectionName;
        }
    }

    ///<inheritdoc/>
    public async Task<(MemoryRecord, double)?> GetNearestMatchAsync(
        string collectionName, 
        Embedding<float> embedding, 
        double minRelevanceScore = 0, 
        bool withEmbedding = false, 
        CancellationToken cancellationToken = default)
    {
        MilvusSearchResult searchResult = await _milvusClient.SearchAsync(
            MilvusSearchParameters.Create(collectionName, EmbeddingFieldName, new[] { EmbeddingFieldName,MetadataFieldName })
            .WithTopK(1),
            cancellationToken: cancellationToken);

        if (searchResult.Results.FieldsData.Any() != true)
        {
            return null;
        }

        double score = searchResult.Results.Scores[0];

        var embeddingField = searchResult.Results.FieldsData[0] as FloatVectorField;
        var metadataField = searchResult.Results.FieldsData[1] as Field<string>;

        return (MemoryRecord.FromJsonMetadata(
            metadataField.Data[0],
            new Embedding<float>(embeddingField.Data[0])),score);
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<(MemoryRecord, double)> GetNearestMatchesAsync(
        string collectionName, 
        Embedding<float> embedding, 
        int limit, 
        double minRelevanceScore = 0, 
        bool withEmbeddings = false, 
        CancellationToken cancellationToken = default)
    {
        MilvusSearchResult searchResult = await _milvusClient.SearchAsync(
            MilvusSearchParameters.Create(collectionName, EmbeddingFieldName, new[] { EmbeddingFieldName, MetadataFieldName })
            .WithTopK(limit),
            cancellationToken: cancellationToken);

        if (searchResult.Results.FieldsData.Any() != true)
        {
            yield break;
        }

        for (int i = 0; i < searchResult.Results.FieldsData.Count; i++)
        {
            double score = searchResult.Results.Scores[i];
            if (score < minRelevanceScore)
            {
                continue;
            }

            var embeddingField = searchResult.Results.FieldsData[i*2] as FloatVectorField;
            var metadataField = searchResult.Results.FieldsData[2*i+1] as Field<string>;

            yield return (MemoryRecord.FromJsonMetadata(
                metadataField.Data[0],
                new Embedding<float>(embeddingField.Data[0])), score);
        }
    }

    ///<inheritdoc/>
    public async Task RemoveAsync(
        string collectionName, 
        string key, 
        CancellationToken cancellationToken = default)
    {
        await this._milvusClient.DeleteAsync(
            collectionName,
            $"{IdFieldName} = {key}",
            cancellationToken:cancellationToken);
    }

    ///<inheritdoc/>
    public async Task RemoveBatchAsync(
        string collectionName, 
        IEnumerable<string> keys, 
        CancellationToken cancellationToken = default)
    {
        StringBuilder stringBuilder = GetKeyGroup(keys);

        await this._milvusClient.DeleteAsync(
            collectionName,
            $"{EmbeddingFieldName} in [{stringBuilder.ToString()}]",
            cancellationToken: cancellationToken);
    }

    ///<inheritdoc/>
    public async Task<string> UpsertAsync(
        string collectionName, 
        MemoryRecord record, 
        CancellationToken cancellationToken = default)
    {
        MilvusMutationResult insertResult = await this._milvusClient.InsertAsync(
            collectionName,
            new[] { 
                ToIdField(record), 
                ToFloatField(record),
                ToMetadataField(record)},
            cancellationToken: cancellationToken);

        return insertResult.Ids.IdField.StrId.Data[0];
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<string> UpsertBatchAsync(
        string collectionName, 
        IEnumerable<MemoryRecord> records, 
        CancellationToken cancellationToken = default)
    {
        if (records?.Any() != true)
        {
            yield break;
        }

        MilvusMutationResult insertResult = await this._milvusClient.InsertAsync(
            collectionName,
            new[] { 
                ToIdField(records), 
                ToFloatField(records),
                ToMetadataField(records)},
            cancellationToken: cancellationToken);

        foreach (var id in insertResult.Ids.IdField.StrId.Data)
        {
            yield return id;
        }
    }

    #region Private ===============================================================================
    private async Task BuildIndexAndLoad(string collectionName,string fieldName,CancellationToken cancellationToken = default)
    {
        IndexState indexSate = await _milvusClient.GetIndexState(collectionName,fieldName, cancellationToken);

        if (indexSate == IndexState.None)
        {
            await _milvusClient.CreateIndexAsync(
                collectionName,
                fieldName,
                Constants.DEFAULT_INDEX_NAME,
                _zillizCloud ? MilvusIndexType.AUTOINDEX : MilvusIndexType.IVF_FLAT,
                MilvusMetricType.IP,
                new Dictionary<string, string>(),
                cancellationToken);
        }

        await _milvusClient.LoadCollectionAsync(collectionName);
    }

    private static StringBuilder GetKeyGroup(IEnumerable<string> keys)
    {
        StringBuilder stringBuilder = new();
        foreach (var key in keys)
        {
            if (stringBuilder.Length > 0)
            {
                stringBuilder.Append(",");
            }
            stringBuilder.Append($"\"{key}\"");
        }

        return stringBuilder;
    }

    private Field ToIdField(MemoryRecord record)
    {
        return Field.Create<string>(IdFieldName, new[] { record.Metadata.Id });
    }

    private Field ToMetadataField(MemoryRecord record)
    {
        return Field.CreateVarChar(IdFieldName, new[] { record.GetSerializedMetadata() });
    }

    private Field ToFloatField(MemoryRecord record)
    {
        return Field.CreateFloatVector(EmbeddingFieldName, new List<List<float>> { record.Embedding.Vector.ToList() });
    }

    private Field ToFloatField(IEnumerable<MemoryRecord> records)
    {
        return Field.Create<string>(IdFieldName, records.Select(m => m.Metadata.Id).ToList());
    }

    private Field ToIdField(IEnumerable<MemoryRecord> records)
    {
        return Field.CreateFloatVector(EmbeddingFieldName, records.Select(m => m.Embedding.Vector.ToList()).ToList());
    }

    private Field ToMetadataField(IEnumerable<MemoryRecord> record)
    {
        return Field.CreateVarChar(IdFieldName, record.Select(m => m.GetSerializedMetadata()).ToList());
    }
    #endregion
}
