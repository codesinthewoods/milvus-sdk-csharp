using Xunit;
using Microsoft.SemanticKernel.AI.Embeddings;
using Microsoft.SemanticKernel.Memory;
using IO.Milvus;
using IO.Milvus.Client;
using Connectors.Memory.MilvusTests;

namespace Connectors.Memory.Milvus.Tests;

public partial class MilvusMemoryStoreTests
{
    private readonly string _collectionName = "test2";
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

    [Theory]
    [ClassData(typeof(TestClients))]
    public async Task ConnectionCanBeInitialized(IMilvusClient milvusClient)
    {
        MilvusHealthState healthState = await milvusClient.HealthAsync(default);
        Assert.True(healthState.IsHealthy,healthState.ErrorMsg);

        var db = new MilvusMemoryStore(milvusClient, 3, MilvusIndexType.AUTOINDEX);
        Assert.NotNull(db);
    }

    [Theory]
    [ClassData(typeof(TestClients))]
    public async Task ConnectorTest(IMilvusClient milvusClient)
    {
        var db = new MilvusMemoryStore(milvusClient, 3, MilvusIndexType.AUTOINDEX);

        //Clear previous exist collection.
        bool collectionExist = await db.DoesCollectionExistAsync(this._collectionName);
        if (collectionExist)
        {
            await db.DeleteCollectionAsync(this._collectionName);
        }

        //Create collection.
        await db.CreateCollectionAsync(this._collectionName);
        collectionExist = await db.DoesCollectionExistAsync(this._collectionName);
        Assert.True(collectionExist);

        //Wait for collection loaded
        await Task.Delay(TimeSpan.FromSeconds(10));

        //Validate collection.
        var collections = await db.GetCollectionsAsync().ToListAsync();
        Assert.Contains(this._collectionName, collections);

        //Create a record
        var memoryRecord = CreateRecords().First();

        //Insert record
        string id = await db.UpsertAsync(this._collectionName, memoryRecord);
        await Task.Delay(1000);

        //Query record
        MemoryRecord? returnedRecord = await db.GetAsync(this._collectionName, id);
        Assert.NotNull(returnedRecord);
        Assert.True(returnedRecord.Metadata.Id == this._id);
        
        //Delete record 
        await db.RemoveAsync(this._collectionName, id);
        await Task.Delay(1000);

        //Check
        returnedRecord = await db.GetAsync(this._collectionName, id);
        Assert.Null(returnedRecord);

        //Batch insert
        IAsyncEnumerable<string> ids = db.UpsertBatchAsync(this._collectionName, CreateRecords());
        Assert.NotEmpty(ids.ToEnumerable());
        Assert.Equal(3, ids.CountAsync().Result);
        await Task.Delay(1000);

        //Query record
        returnedRecord = await db.GetAsync(this._collectionName, id);
        Assert.NotNull(returnedRecord);
        Assert.True(returnedRecord.Metadata.Id == this._id);

        var records = db.GetBatchAsync(this._collectionName, new[] {this._id,this._id2}).ToEnumerable();
        Assert.Equal(2, records.Count());
        Assert.Contains(records, r => r.Metadata.Id == this._id);
        Assert.Contains(records, r => r.Metadata.Id == this._id2);

        //Get nearest
        (MemoryRecord, double)? match = await db.GetNearestMatchAsync(this._collectionName, _embedding);
        Assert.NotNull(match);

        IAsyncEnumerable<(MemoryRecord, double)> matches = db.GetNearestMatchesAsync(this._collectionName, _embedding, 2);
        Assert.NotNull(match);

        //Delete record 
        await db.RemoveAsync(this._collectionName, id);
        await Task.Delay(1000);

        //Check
        returnedRecord = await db.GetAsync(this._collectionName, id);
        Assert.Null(returnedRecord);

        //Delete collection
        //await db.DeleteCollectionAsync(this._collectionName);
    }

    private IEnumerable<MemoryRecord> CreateRecords()
    {
        var e = Normalize(_embedding);
        var e2 = Normalize(_embedding2);
        var e3 = Normalize(_embedding3);

        yield return MemoryRecord.LocalRecord(
            id: this._id,
            text: this._text,
            description: this._description,
            embedding: e);

        yield return MemoryRecord.LocalRecord(
            id: this._id2,
            text: this._text2,
            description: this._description2,
            embedding: e2);

        yield return MemoryRecord.LocalRecord(
            id: this._id3,
            text: this._text3,
            description: this._description3,
            embedding: e3);
    }

    private Embedding<float> Normalize(Embedding<float> embedding)
    {
        return embedding;
        //List<float> data = embedding.Vector.ToList();
        //Vector3 vector = Vector3.Normalize((new Vector3(data[0], data[1], data[2])));
        //return new Embedding<float>(new[] { vector.X, vector.Y, vector.Z });
    }
}