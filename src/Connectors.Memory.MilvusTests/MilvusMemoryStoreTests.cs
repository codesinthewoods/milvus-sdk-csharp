using Xunit;
using IO.Milvus.Client.gRPC;
using Microsoft.SemanticKernel.AI.Embeddings;
using System.Reflection.Emit;
using Microsoft.SemanticKernel.Memory;
using IO.Milvus;

namespace Connectors.Memory.Milvus.Tests;

public class MilvusMemoryStoreTests
{
    private readonly string _collectionName = "test";
    private readonly string _id = "Id";
    private readonly string _id2 = "Id2";
    private readonly string _id3 = "Id3";
    private readonly string _text = "text";
    private readonly string _text2 = "text2";
    private readonly string _text3 = "text3";
    private readonly string _description = "description";
    private readonly string _description2 = "description2";
    private readonly string _description3 = "description3";
    private readonly Embedding<float> _embedding = new Embedding<float>(new float[] { 1, 1, 1 });
    private readonly Embedding<float> _embedding2 = new Embedding<float>(new float[] { 2, 2, 2 });
    private readonly Embedding<float> _embedding3 = new Embedding<float>(new float[] { 3, 3, 3 });

    [Fact()]
    public async Task<IMemoryStore> ConnectionCanBeInitialized()
    {
        var milvusClient = new MilvusGrpcClient("https://in01-840db488d50733e.aws-us-west-2.vectordb.zillizcloud.com", 19535, "db_admin", "Milvus-CSharp-SDK");
        MilvusHealthState healthState = await milvusClient.HealthAsync(default);
        Assert.True(healthState.IsHealthy,healthState.ErrorMsg);

        var db = new MilvusMemoryStore(milvusClient, 3,true);

        Assert.NotNull(db);
        return db;
    }

    [Fact()]
    public async Task CollectionAsyncTest()
    {
        IMemoryStore db = await ConnectionCanBeInitialized();
        await db.CreateCollectionAsync(_collectionName);

        bool result = await db.DoesCollectionExistAsync(_collectionName);
        Assert.True(result);

        var collections = await db.GetCollectionsAsync().ToListAsync();
        Assert.Contains(_collectionName, collections);

        await db.DeleteCollectionAsync(_collectionName);

        result = await db.DoesCollectionExistAsync(_collectionName);
        Assert.False(result);
    }

    [Fact()]
    public async Task GetAsyncTest()
    {
        IMemoryStore db = await ConnectionCanBeInitialized();
        bool result = await db.DoesCollectionExistAsync(_collectionName);
        if (result)
        {
            await db.DeleteCollectionAsync(_collectionName);
        }
        result = await db.DoesCollectionExistAsync(_collectionName);
        Assert.False(result);
        await db.CreateCollectionAsync(_collectionName);

        result = await db.DoesCollectionExistAsync(_collectionName);
        Assert.True(result);

        var memoryRecord = MemoryRecord.LocalRecord(
            id: this._id,
            text: this._text,
            description: this._description,
            embedding: this._embedding);

        string id = await db.UpsertAsync(_collectionName, memoryRecord);

        await Task.Delay(5000);

        var returnedRecord = await db.GetAsync(_collectionName, id);
        Assert.NotNull(returnedRecord);
    }

    [Fact()]
    public void GetBatchAsyncTest()
    {
        Assert.True(false, "This test needs an implementation");
    }

    [Fact()]
    public void GetCollectionsAsyncTest()
    {
        Assert.True(false, "This test needs an implementation");
    }

    [Fact()]
    public void GetNearestMatchAsyncTest()
    {
        Assert.True(false, "This test needs an implementation");
    }

    [Fact()]
    public void GetNearestMatchesAsyncTest()
    {
        Assert.True(false, "This test needs an implementation");
    }

    [Fact()]
    public void RemoveAsyncTest()
    {
        Assert.True(false, "This test needs an implementation");
    }

    [Fact()]
    public void RemoveBatchAsyncTest()
    {
        Assert.True(false, "This test needs an implementation");
    }

    [Fact()]
    public void UpsertAsyncTest()
    {
        Assert.True(false, "This test needs an implementation");
    }

    [Fact()]
    public void UpsertBatchAsyncTest()
    {
        Assert.True(false, "This test needs an implementation");
    }
}