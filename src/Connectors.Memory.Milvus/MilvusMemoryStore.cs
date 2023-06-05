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
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
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

    #region Ctor
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
        this._zillizCloud = zillizCloud;
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
        this._zillizCloud = zillizCloud;
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
        this._zillizCloud = zillizCloud;
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
        this._zillizCloud = zillizCloud;
        this._log = logger;
        this._milvusClient = new MilvusRestClient(host, port, userName, password, log: logger,httpClient:httpClient);
    }
    #endregion

    ///<inheritdoc/>
    public async Task CreateCollectionAsync(
        string collectionName, 
        CancellationToken cancellationToken = default)
    {
        if (!await this._milvusClient.HasCollectionAsync(
            collectionName,
            cancellationToken:cancellationToken))
        {
            //Create collection
            await this._milvusClient.CreateCollectionAsync(collectionName,
                new FieldType[] { 
                    FieldType.CreateVarchar(IdFieldName,maxLength: 100,isPrimaryKey: true),
                    FieldType.CreateFloatVector(EmbeddingFieldName,_vectorSize),
                    FieldType.CreateVarchar(MetadataFieldName, maxLength: 1000) },
                cancellationToken: cancellationToken
                );

            //Create Index
            await this._milvusClient.CreateIndexAsync(
                collectionName,
                EmbeddingFieldName, 
                Constants.DEFAULT_INDEX_NAME,
                _zillizCloud ? MilvusIndexType.AUTOINDEX : MilvusIndexType.IVF_FLAT,
                MilvusMetricType.IP,
                new Dictionary<string, string> { { "nlist", "1024" } },
                cancellationToken: cancellationToken);

            //Load Collection
            await this._milvusClient.LoadCollectionAsync(collectionName);
        }
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
        string expr = $"{IdFieldName} in [\"{key}\"]";
        MilvusQueryResult result = await this._milvusClient.QueryAsync(collectionName,
            expr,
            new[] {MetadataFieldName, EmbeddingFieldName },
            cancellationToken:cancellationToken);

        if (result.FieldsData?.Any() != true || result.FieldsData.First().RowCount == 0)
        {
            return null;
        }

        var metadataField = result.FieldsData.First(p => p.FieldName == MetadataFieldName) as Field<string>;
        var embeddingField = result.FieldsData.First(p => p.FieldName == EmbeddingFieldName) as FloatVectorField;

        return MemoryRecord.FromJsonMetadata(
            metadataField.Data[0],
            new Embedding<float>(embeddingField.Data[0]));
    }

    public async IAsyncEnumerable<MemoryRecord> GetBatchAsync(
        string collectionName, 
        IEnumerable<string> keys, 
        bool withEmbeddings = false,
        [EnumeratorCancellation]CancellationToken cancellationToken = default)
    {
        var keyList = keys.ToList();
        var keyGroup = GetKeyGroup(keyList);
        var expr = $"{IdFieldName} in [{keyGroup}]";

        MilvusQueryResult result = await this._milvusClient.QueryAsync(collectionName,
            expr,
            new[] { MetadataFieldName,EmbeddingFieldName},
            cancellationToken: cancellationToken);

        if (result.FieldsData?.Any() != true)
        {
            yield break;
        }

        var metadataField = result.FieldsData.First(p => p.FieldName == MetadataFieldName) as Field<string>;
        var embeddingField = result.FieldsData.First(p => p.FieldName == EmbeddingFieldName) as FloatVectorField;

        for (int i = 0; i < metadataField.RowCount; i++)
        {
            yield return MemoryRecord.FromJsonMetadata(
                metadataField.Data[i],
                new Embedding<float>(embeddingField.Data[i]));
        }
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<string> GetCollectionsAsync([EnumeratorCancellation]CancellationToken cancellationToken = default)
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
        //Milvus does not support vector field in out fields
        MilvusSearchResult searchResult = await _milvusClient.SearchAsync(
            MilvusSearchParameters.Create(collectionName, EmbeddingFieldName, new[] { MetadataFieldName })
            .WithConsistencyLevel(MilvusConsistencyLevel.Strong)
            .WithTopK(topK: 1)
            .WithVectors(new[] { embedding.Vector.ToList() })
            .WithMetricType(MilvusMetricType.IP)
            .WithParameter("nprobe", "10")
            .WithParameter("offset", "5"),
            cancellationToken: cancellationToken);

        if (searchResult.Results.FieldsData.Any() != true || searchResult.Results.FieldsData.First().RowCount == 0)
        {
            return null;
        }

        double score = searchResult.Results.Scores[0];
        if (score < minRelevanceScore)
        {
            return null;
        }

        var metadataField = searchResult.Results.FieldsData[0] as Field<string>;

        if (withEmbedding)
        {
            var metadata = JsonSerializer.Deserialize<MemoryRecordMetadata>(metadataField.Data[0]);
            return (await GetAsync(collectionName, metadata.Id,withEmbedding), score);
        }
        else
        {
            return (MemoryRecord.FromJsonMetadata(
                metadataField.Data[0],
                null), score);
        }
    }

    ///<inheritdoc/>
    public async IAsyncEnumerable<(MemoryRecord, double)> GetNearestMatchesAsync(
        string collectionName, 
        Embedding<float> embedding, 
        int limit, 
        double minRelevanceScore = 0, 
        bool withEmbeddings = false, 
        [EnumeratorCancellation]CancellationToken cancellationToken = default)
    {
        //Milvus does not support vector field in out fields
        MilvusSearchResult searchResult = await _milvusClient.SearchAsync(
            MilvusSearchParameters.Create(collectionName, EmbeddingFieldName, new[] { MetadataFieldName })
            .WithConsistencyLevel(MilvusConsistencyLevel.Strong)
            .WithTopK(topK: limit)
            .WithVectors(new[] { embedding.Vector.ToList() })
            .WithMetricType(MilvusMetricType.IP)
            .WithParameter("nprobe", "10")
            .WithParameter("offset", "5"),
            cancellationToken: cancellationToken);

        if (searchResult.Results.FieldsData.Any() != true || searchResult.Results.FieldsData.First().RowCount == 0)
        {
            yield break;
        }

        var metadataField = searchResult.Results.FieldsData[0] as Field<string>;

        for (int i = 0; i < metadataField.RowCount; i++)
        {
            double score = searchResult.Results.Scores[i];
            if (score < minRelevanceScore)
            {
                continue;
            }

            if (withEmbeddings)
            {
                var metadata = JsonSerializer.Deserialize<MemoryRecordMetadata>(metadataField.Data[0]);
                yield return (await GetAsync(collectionName, metadata.Id, withEmbeddings), score);
            }
            else
            {
                yield return (MemoryRecord.FromJsonMetadata(
                    metadataField.Data[i],
                    null), score);
            }
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
            $"{IdFieldName} in [\"{key}\"]",
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
        [EnumeratorCancellation]CancellationToken cancellationToken = default)
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
    private static StringBuilder GetKeyGroup(IEnumerable<string> keys)
    {
        StringBuilder stringBuilder = new();
        foreach (var key in keys)
        {
            if (stringBuilder.Length > 0)
            {
                stringBuilder.Append(", ");
            }
            stringBuilder.Append($"\"{key}\"");
        }

        return stringBuilder;
    }

    private Field ToIdField(MemoryRecord record)
    {
        return Field.Create<string>(IdFieldName, new[] { record.Metadata.Id });
    }

    private Field ToIdField(IEnumerable<MemoryRecord> records)
    {
        return Field.Create<string>(IdFieldName, records.Select(m => m.Metadata.Id).ToList());
    }

    private Field ToFloatField(MemoryRecord record)
    {
        return Field.CreateFloatVector(EmbeddingFieldName, new List<List<float>> { record.Embedding.Vector.ToList() });
    }

    private Field ToFloatField(IEnumerable<MemoryRecord> records)
    {
        return Field.CreateFloatVector(EmbeddingFieldName, records.Select(m => m.Embedding.Vector.ToList()).ToList());
    }

    private Field ToMetadataField(MemoryRecord record)
    {
        return Field.CreateVarChar(MetadataFieldName, new[] { record.GetSerializedMetadata() });
    }

    private Field ToMetadataField(IEnumerable<MemoryRecord> record)
    {
        return Field.CreateVarChar(MetadataFieldName, record.Select(m => m.GetSerializedMetadata()).ToList());
    }
    #endregion
}
